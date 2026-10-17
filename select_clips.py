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


DATASET_OPTIONS = {
    "dev-clean": ("clean", "validation"),
    "dev-other": ("other", "validation"),
    "test-clean": ("clean", "test"),
    "test-other": ("other", "test"),
}


def load_clips(dataset_name):
    """Read clip IDs and transcripts without fetching audio files."""
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, help="new selection CSV")
    args = parser.parse_args()
    if args.num_clips < 1:
        parser.error("--num-clips must be positive")
    if args.output and not args.speakers:
        parser.error("--output requires --speakers")
    output_file = args.output or project_dir / "outputs" / f"{args.dataset}_selected_clips.csv"

    try:
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
        selected = select_clips(clips, args.speakers, args.num_clips, args.seed)
        save_selection(selected, output_file)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Error: {error}\n")

    print(f"\nSelected: {len(selected)} clips | Seed: {args.seed}")
    for speaker, count in sorted(Counter(clip["speaker_id"] for clip in selected).items()):
        print(f"Speaker {speaker}: {count} clips")
    print(f"Saved to {output_file}")


if __name__ == "__main__":
    main()
