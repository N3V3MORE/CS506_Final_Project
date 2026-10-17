"""Choose a balanced LibriSpeech clip list before transcription.

List speakers: python select_clips.py
Example: python select_clips.py --speakers 3660 --num-clips 10

Reads metadata only, with seed 42 by default. Speaker counts differ by at most
one. Saves a selection CSV under outputs/.
Run transcribe_selection.py to transcribe the saved list.
"""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import random
from urllib.parse import urlencode
from urllib.request import urlopen
from urllib.error import HTTPError, URLError
from time import sleep


DATASET_OPTIONS = {
    "dev-clean": ("clean", "validation"),
    "dev-other": ("other", "validation"),
    "test-clean": ("clean", "test"),
    "test-other": ("other", "test"),
    "train-other-500": ("other", "train.500"),
}


def load_clips(dataset_name):
    """Read clip IDs and transcripts without fetching audio files."""
    if dataset_name == "train-other-500":
        raise ValueError("Use selected-speaker metadata for train-other-500.")
    config, split = DATASET_OPTIONS[dataset_name]
    clips = []
    while True:
        query = urlencode({
            "dataset": "openslr/librispeech_asr",
            "config": config,
            "split": split,
            "offset": len(clips),
            "length": 100,
        })
        with urlopen(f"https://datasets-server.huggingface.co/rows?{query}", timeout=30) as response:
            data = json.load(response)
        if data["partial"] or not data["rows"]:
            raise ValueError("Hugging Face returned incomplete clip metadata.")
        for item in data["rows"]:
            row = item["row"]
            clips.append({
                "dataset": dataset_name,
                "audio_id": row["id"],
                "speaker_id": str(row["speaker_id"]),
                "chapter_id": str(row["chapter_id"]),
                "reference_text": row["text"],
            })
        if len(clips) == data["num_rows_total"]:
            return clips
        if len(clips) > data["num_rows_total"]:
            raise ValueError("Hugging Face returned inconsistent clip counts.")


def request_metadata(url):
    """Retry brief server failures; rate limits stop the run immediately."""
    for attempt in range(3):
        try:
            with urlopen(url, timeout=30) as response:
                return json.load(response)
        except HTTPError as error:
            if error.code == 429:
                retry_after = error.headers.get("Retry-After", "unspecified")
                raise RuntimeError(
                    f"Dataset server rate limit reached. Retry-After: {retry_after}. "
                    "Stop requests until that interval has passed."
                ) from error
            if error.code not in (500, 502, 503, 504) or attempt == 2:
                raise
        except (URLError, TimeoutError):
            if attempt == 2:
                raise
        sleep(2 ** attempt)


def load_speaker_metadata(dataset_name, speaker_id):
    """Fetch one speaker's metadata, never a complete training split."""
    speaker_id = str(int(speaker_id))
    config, split = DATASET_OPTIONS[dataset_name]
    recordings = []
    seen_ids = set()
    expected_total = None
    while True:
        query = urlencode({
            "dataset": "openslr/librispeech_asr", "config": config, "split": split,
            "where": f'"speaker_id"={speaker_id}', "offset": len(recordings), "length": 100,
        })
        data = request_metadata(f"https://datasets-server.huggingface.co/filter?{query}")
        total = data["num_rows_total"]
        if data["partial"] or (expected_total is not None and total != expected_total):
            raise ValueError("Hugging Face returned incomplete speaker metadata.")
        expected_total = total
        if not data["rows"] and total != 0:
            raise ValueError("Hugging Face returned an empty speaker metadata page.")
        for item in data["rows"]:
            row = item["row"]
            if str(row["speaker_id"]) != speaker_id or row["id"] in seen_ids:
                raise ValueError("Speaker metadata contains a wrong speaker or duplicate ID.")
            seen_ids.add(row["id"])
            recordings.append(row)
        if len(recordings) == total:
            return recordings
        if len(recordings) > total:
            raise ValueError("Hugging Face returned inconsistent speaker counts.")


def read_prior_ids(paths):
    """Accept saved selection/results CSVs and the earlier JSON manifests."""
    speakers, audio_ids = set(), set()
    for path in paths:
        path = Path(path)
        with path.open(encoding="utf-8-sig", newline="") as source:
            if path.suffix.lower() == ".json":
                rows = json.load(source)["clips"]
            else:
                rows = list(csv.DictReader(source, strict=True))
        if not rows:
            raise ValueError(f"No historical clips in {path}")
        for row in rows:
            audio_id = row.get("audio_id") or row.get("id")
            if not audio_id or not str(row.get("speaker_id", "")).strip():
                raise ValueError(f"Missing historical audio/speaker ID in {path}")
            speakers.add(str(int(row["speaker_id"])))
            audio_ids.add(audio_id)
    return speakers, audio_ids


def read_candidate_speakers(path):
    """Read LibriSpeech SPEAKERS.TXT or one numeric speaker ID per line."""
    speakers = set()
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith((";", "#")):
            continue
        parts = [part.strip() for part in line.split("|")]
        if len(parts) > 1 and (len(parts) < 3 or parts[2] != "train-other-500"):
            continue
        speakers.add(str(int(parts[0])))
    if not speakers:
        raise ValueError("Candidate speaker file contains no train-other-500 speakers.")
    return sorted(speakers, key=int)


def select_fresh_clips(candidate_speakers, excluded_speakers, excluded_audio_ids,
                       speaker_count=80, clips_per_speaker=25, seed=506):
    if speaker_count < 2 or clips_per_speaker < 1:
        raise ValueError("Request at least two speakers and one clip per speaker.")
    candidates = sorted({str(int(s)) for s in candidate_speakers}, key=int)
    excluded_speakers = {str(int(s)) for s in excluded_speakers}
    candidates = [s for s in candidates if s not in excluded_speakers]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    selected = []
    eligible_speakers = 0
    for speaker in candidates:
        recordings = load_speaker_metadata("train-other-500", speaker)
        recordings = [row for row in recordings if row["id"] not in excluded_audio_ids]
        if len(recordings) < clips_per_speaker:
            continue
        recordings.sort(key=lambda row: row["id"])
        for row in rng.sample(recordings, clips_per_speaker):
            selected.append({
                "dataset": "train-other-500", "audio_id": row["id"],
                "speaker_id": str(row["speaker_id"]), "chapter_id": str(row["chapter_id"]),
                "reference_text": row["text"],
            })
        eligible_speakers += 1
        if eligible_speakers == speaker_count:
            rng.shuffle(selected)
            return selected
    raise ValueError(f"Only {eligible_speakers} eligible unused speakers; need {speaker_count}.")


def select_clips(clips, speaker_ids, num_clips=1000, seed=42):
    speakers = sorted({str(speaker).strip() for speaker in speaker_ids})
    if not speakers or "" in speakers:
        raise ValueError("Supply at least one speaker ID.")
    if num_clips < len(speakers):
        raise ValueError("Request at least one clip per selected speaker.")

    clips_by_speaker = {speaker: [] for speaker in speakers}
    seen_ids = set()
    for clip in clips:
        speaker = clip["speaker_id"]
        if speaker not in clips_by_speaker:
            continue
        if clip["audio_id"] in seen_ids:
            raise ValueError(f"Duplicate audio ID: {clip['audio_id']}")
        seen_ids.add(clip["audio_id"])
        clips_by_speaker[speaker].append(clip)

    missing = [speaker for speaker in speakers if not clips_by_speaker[speaker]]
    if missing:
        raise ValueError(f"Speaker IDs not found in this split: {', '.join(missing)}")

    per_speaker, remainder = divmod(num_clips, len(speakers))
    extra_candidates = [
        speaker for speaker in speakers if len(clips_by_speaker[speaker]) > per_speaker
    ]
    if (any(len(rows) < per_speaker for rows in clips_by_speaker.values())
            or len(extra_candidates) < remainder):
        available = ", ".join(f"{speaker}: {len(rows)}" for speaker, rows in clips_by_speaker.items())
        raise ValueError(f"Cannot select {num_clips} clips evenly. Available per speaker: {available}")

    rng = random.Random(seed)
    extra_speakers = set(rng.sample(extra_candidates, remainder))
    selected = []
    for speaker in speakers:
        candidates = sorted(clips_by_speaker[speaker], key=lambda clip: clip["audio_id"])
        count = per_speaker + (speaker in extra_speakers)
        selected.extend(rng.sample(candidates, count))
    rng.shuffle(selected)
    return selected


def save_selection(clips, output_file):
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("x", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=[
            "dataset", "audio_id", "speaker_id", "chapter_id", "reference_text",
        ])
        writer.writeheader()
        writer.writerows(clips)


def main():
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", choices=DATASET_OPTIONS, default="dev-other")
    parser.add_argument("--speakers", nargs="+", metavar="ID",
                        help="speaker IDs to include; omit to list available counts")
    parser.add_argument("--num-clips", type=int, default=1000)
    parser.add_argument("--seed", type=int, help="default 42; fresh study uses 506")
    parser.add_argument("--output", type=Path, help="new selection CSV")
    parser.add_argument("--candidate-speakers", type=Path,
                        help="fresh study: SPEAKERS.TXT or numeric candidate IDs")
    parser.add_argument("--exclude", nargs="+", type=Path,
                        help="historical CSVs/JSON manifests to exclude")
    parser.add_argument("--speaker-count", type=int, default=80)
    parser.add_argument("--clips-per-speaker", type=int, default=25)
    args = parser.parse_args()
    seed = args.seed if args.seed is not None else (506 if args.candidate_speakers else 42)
    if args.num_clips < 1:
        parser.error("--num-clips must be positive")
    if args.output and not (args.speakers or args.candidate_speakers):
        parser.error("--output requires --speakers or --candidate-speakers")
    output_file = args.output or project_dir / "outputs" / f"{args.dataset}_selected_clips.csv"

    if args.dataset == "train-other-500" and not args.candidate_speakers:
        parser.error("train-other-500 requires --candidate-speakers; full-split scans are disabled")
    if args.candidate_speakers and (args.dataset != "train-other-500" or not args.exclude or args.speakers):
        parser.error("fresh selection requires --dataset train-other-500 and --exclude, without --speakers")
    try:
        if args.candidate_speakers:
            if output_file.exists():
                raise FileExistsError(f"Output already exists: {output_file}. Choose another --output.")
            excluded_speakers, excluded_ids = read_prior_ids(args.exclude)
            candidates = read_candidate_speakers(args.candidate_speakers)
            selected = select_fresh_clips(
                candidates, excluded_speakers, excluded_ids, args.speaker_count,
                args.clips_per_speaker, seed,
            )
            save_selection(selected, output_file)
            print(f"Saved {len(selected)} clips from {args.speaker_count} unused speakers to {output_file}")
            return
        if args.speakers and output_file.exists():
            raise FileExistsError(f"Output already exists: {output_file}. Choose another --output.")
        print(f"Reading {args.dataset} metadata (no audio downloads)...", flush=True)
        clips = load_clips(args.dataset)
        available = Counter(clip["speaker_id"] for clip in clips)
        print(f"Available: {len(clips)} clips | {len(available)} speakers")
        for speaker, count in sorted(available.items()):
            print(f"Speaker {speaker}: {count} clips")
        if not args.speakers:
            print("Use --speakers ID ID ... to save a selection.")
            return
        selected = select_clips(clips, args.speakers, args.num_clips, seed)
        save_selection(selected, output_file)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")

    print(f"\nSelected: {len(selected)} clips | Seed: {seed}")
    for speaker, count in sorted(Counter(clip["speaker_id"] for clip in selected).items()):
        print(f"Speaker {speaker}: {count} clips")
    print(f"Saved to {output_file}")


if __name__ == "__main__":
    main()
