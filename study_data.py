"""Read selected LibriSpeech metadata/audio through bounded Parquet ranges.

This bypasses the Dataset Viewer. Metadata reads exclude audio columns.
Audio is fetched by selected row group (up to 100 recordings), with an explicit
byte budget; only selected FLAC files are retained. Requires pyarrow.
"""

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import io
import hashlib
import json
from pathlib import Path
import re
from threading import Lock
from urllib.parse import quote
from urllib.request import Request, urlopen
from urllib.error import HTTPError

from select_clips import DATASET_OPTIONS


class RangeFile(io.RawIOBase):
    """The seekable file interface PyArrow needs, without whole-file reads."""

    def __init__(self, url, size, budget):
        self.url, self.size, self.budget = url, size, budget
        self.position = 0
        self.footer_sha256 = None

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if position < 0 or position > self.size:
            raise ValueError("Parquet seek outside file bounds.")
        self.position = position
        return position

    def read(self, length=-1):
        if length < 0:
            raise ValueError("Whole-file reads are disabled.")
        length = min(length, self.size - self.position)
        if not length:
            return b""
        if length > 64 * 1024 ** 2:
            raise ValueError("Parquet range exceeds the 64 MiB request limit.")
        with self.budget["lock"]:
            if self.budget.get("stopped"):
                raise RuntimeError("A previous Parquet request failed; remaining requests stopped.")
            if self.budget["used"] + length > self.budget["limit"]:
                raise ValueError("Parquet download byte budget exceeded.")
            self.budget["used"] += length
        start, end = self.position, self.position + length - 1
        request = Request(self.url, headers={"Range": f"bytes={start}-{end}"})
        try:
            with urlopen(request, timeout=30) as response:
                expected = f"bytes {start}-{end}/{self.size}"
                if response.status != 206 or response.headers.get("Content-Range") != expected:
                    raise ValueError("Server did not honor the exact HTTP byte range; download stopped.")
                data = response.read(length + 1)
            if len(data) != length:
                raise ValueError("Parquet range response length mismatch.")
        except Exception as error:
            with self.budget["lock"]:
                self.budget["stopped"] = True
            if isinstance(error, HTTPError) and error.code == 429:
                retry_after = (error.headers or {}).get("Retry-After", "unspecified")
                raise RuntimeError(f"Parquet server rate limit reached. Retry-After: {retry_after}. "
                                   "Requests stopped; wait before retrying.") from error
            raise
        if end == self.size - 1 and len(data) >= 8 and data[-4:] == b"PAR1":
            footer_size = int.from_bytes(data[-8:-4], "little") + 8
            if footer_size <= len(data):
                self.footer_sha256 = hashlib.sha256(data[-footer_size:]).hexdigest()
        self.position += length
        return data


def download_budget(max_bytes):
    return {"limit": max_bytes, "used": 0, "lock": Lock()}


def parquet_sources(dataset):
    config, split = DATASET_OPTIONS[dataset]
    with urlopen("https://huggingface.co/api/datasets/openslr/librispeech_asr", timeout=30) as response:
        revision = json.load(response)["sha"]
    path = f"{config}/{split}"
    url = f"https://huggingface.co/api/datasets/openslr/librispeech_asr/tree/{revision}/{path}?limit=1000"
    with urlopen(url, timeout=30) as response:
        if response.headers.get("Link"):
            raise ValueError("Shard listing is paginated; inspect it before proceeding.")
        files = json.load(response)
    shards = [{
        "url": f"https://huggingface.co/datasets/openslr/librispeech_asr/resolve/{revision}/{quote(item['path'])}",
        "size": item["size"], "sha256": item.get("lfs", {}).get("oid"),
    } for item in files if item["type"] == "file" and item["path"].endswith(".parquet")]
    if not shards:
        raise ValueError("No Parquet shards found for this split.")
    return {"dataset": dataset, "revision": revision, "shards": shards}


def index_recordings(dataset, speaker_ids, max_bytes=64 * 1024 ** 2):
    import pyarrow.parquet as pq

    source = parquet_sources(dataset)
    speakers = {int(speaker) for speaker in speaker_ids}
    budget = download_budget(max_bytes)

    def read_shard(shard):
        recordings = []
        with RangeFile(shard["url"], shard["size"], budget) as remote:
            parquet = pq.ParquetFile(remote, pre_buffer=False)
            shard["footer_sha256"] = remote.footer_sha256
            for group_index in range(parquet.metadata.num_row_groups):
                group = parquet.metadata.row_group(group_index)
                columns = {group.column(i).path_in_schema: group.column(i) for i in range(group.num_columns)}
                stats = columns["speaker_id"].statistics
                if stats and stats.has_min_max and not any(stats.min <= speaker <= stats.max for speaker in speakers):
                    continue
                audio_bytes = sum(column.total_compressed_size for name, column in columns.items()
                                  if name.startswith("audio."))
                table = parquet.read_row_group(group_index, columns=["id", "speaker_id", "chapter_id", "text"],
                                               use_threads=False)
                for index, row in enumerate(table.to_pylist()):
                    if int(row["speaker_id"]) in speakers:
                        recordings.append(dict(row, shard_url=shard["url"], shard_size=shard["size"],
                                               row_group=group_index, row_index=index,
                                               audio_group_bytes=audio_bytes))
        return recordings

    with ThreadPoolExecutor(max_workers=4) as workers:
        rows = [row for shard_rows in workers.map(read_shard, source["shards"]) for row in shard_rows]
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate clip IDs in Parquet metadata.")
    source["metadata_bytes_requested"] = budget["used"]
    return rows, source


def selected_groups(selection, index):
    recordings = {row["id"]: row for row in index}
    groups = defaultdict(list)
    for clip in selection:
        row = recordings[clip["audio_id"]]
        if (str(row["speaker_id"]) != clip["speaker_id"]
                or str(row["chapter_id"]) != clip["chapter_id"] or row["text"] != clip["reference_text"]):
            raise ValueError(f"{clip['audio_id']}: Parquet metadata differs from selection.")
        groups[(row["shard_url"], row["row_group"])].append((clip, row))
    return groups


def estimate_audio_bytes(selection, index):
    groups = selected_groups(selection, index)
    return {"row_groups": len(groups), "compressed_audio_bytes": sum(rows[0][1]["audio_group_bytes"]
                                                                   for rows in groups.values())}


def cache_recordings(selection, index, cache_root, max_bytes):
    import pyarrow.parquet as pq

    if len({clip["audio_id"] for clip in selection}) != len(selection):
        raise ValueError("Duplicate selected audio IDs.")
    for clip in selection:
        if not re.fullmatch(r"[0-9]+-[0-9]+-[0-9]+", clip["audio_id"]) or clip["dataset"] not in DATASET_OPTIONS:
            raise ValueError("Unsafe cache filename or dataset.")
    for clip in selection:
        existing = Path(cache_root) / clip["dataset"] / f"{clip['audio_id']}.flac"
        if existing.exists():
            with existing.open("rb") as source:
                if existing.stat().st_size < 42 or source.read(4) != b"fLaC":
                    raise ValueError(f"Invalid existing FLAC cache: {existing}. File left unchanged.")
    groups = selected_groups(selection, index)
    estimate = estimate_audio_bytes(selection, index)
    if estimate["compressed_audio_bytes"] + len(groups) * 65536 > max_bytes:
        raise ValueError("Selected audio row groups exceed the specified byte budget.")
    budget = download_budget(max_bytes)
    saved = 0
    for group_number, ((url, group_index), rows) in enumerate(groups.items(), 1):
        pending = [(clip, row) for clip, row in rows
                   if not (Path(cache_root) / clip["dataset"] / f"{clip['audio_id']}.flac").exists()]
        if not pending:
            continue
        with RangeFile(url, rows[0][1]["shard_size"], budget) as remote:
            parquet = pq.ParquetFile(remote, pre_buffer=False)
            table = parquet.read_row_group(group_index, columns=["id", "text", "audio"], use_threads=False)
            for clip, row in pending:
                item = table.slice(row["row_index"], 1).to_pylist()[0]
                if item["id"] != clip["audio_id"] or item["text"] != clip["reference_text"]:
                    raise ValueError("Audio row ID/reference mismatch.")
                audio = item["audio"]["bytes"]
                if not audio or not audio.startswith(b"fLaC"):
                    raise ValueError(f"{clip['audio_id']}: expected FLAC bytes.")
                output = Path(cache_root) / clip["dataset"] / f"{clip['audio_id']}.flac"
                output.parent.mkdir(parents=True, exist_ok=True)
                partial = output.with_suffix(".flac.part")
                with partial.open("xb") as destination:
                    destination.write(audio)
                if output.exists():
                    raise FileExistsError(f"Cache target already exists: {output}")
                partial.rename(output)
                saved += 1
        print(f"Cached row group {group_number}/{len(groups)} | Saved clips: {saved} | "
              f"Bytes requested: {budget['used']}", flush=True)
    hashes = {}
    for clip in selection:
        output = Path(cache_root) / clip["dataset"] / f"{clip['audio_id']}.flac"
        hashes[clip["audio_id"]] = hashlib.sha256(output.read_bytes()).hexdigest()
    return {"saved": saved, "bytes_requested": budget["used"], "audio_sha256": hashes, **estimate}
