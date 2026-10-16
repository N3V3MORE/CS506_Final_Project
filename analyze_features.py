"""Compare reference sound proportions with saved whole-clip WER.

Run build_features.py first, then:
    python analyze_features.py [features.csv] [--output associations.csv]

Correlations are exploratory: length, word choice, and other sound proportions
are not controlled. They do not locate errors in individual sounds. Console
results require at least five clips containing the sound; the CSV keeps all 39.
Undefined correlations and empty-group averages are left blank.
"""

import argparse
import csv
import math
from pathlib import Path
from statistics import correlation, mean

from build_features import PHONEMES


MIN_CLIPS_WITH_SOUND = 5


def read_features(input_file):
    numeric_columns = ["wer", "num_phonemes", "num_g2p_words", "num_unresolved_words"]
    for phone in PHONEMES:
        numeric_columns.extend([f"{phone}_count", f"{phone}_proportion"])
    required = {"audio_id", "speaker_id", *numeric_columns}
    clips = []
    seen_ids = set()
    with Path(input_file).open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source, strict=True)
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("CSV contains duplicate column names")
        for row_number, row in enumerate(reader, start=2):
            clip_id = row.get("audio_id") or f"CSV row {row_number}"
            try:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError("wrong number of CSV fields")
                if not row["audio_id"].strip() or not row["speaker_id"].strip():
                    raise ValueError("audio_id and speaker_id must be present")
                if row["audio_id"] in seen_ids:
                    raise ValueError("duplicate audio_id")
                for column in numeric_columns:
                    value = float(row[column])
                    if not math.isfinite(value) or value < 0:
                        raise ValueError(f"{column} must be finite and non-negative")
                    if column != "wer" and not column.endswith("_proportion") and not value.is_integer():
                        raise ValueError(f"{column} must be a whole number")
                    row[column] = value
                total = row["num_phonemes"]
                if total == 0 or sum(row[f"{phone}_count"] for phone in PHONEMES) != total:
                    raise ValueError("phoneme counts must sum to a positive num_phonemes")
                for phone in PHONEMES:
                    if not math.isclose(row[f"{phone}_proportion"], row[f"{phone}_count"] / total,
                                        rel_tol=1e-9, abs_tol=1e-9):
                        raise ValueError(f"{phone}_proportion does not match its count")
            except ValueError as error:
                raise ValueError(f"{clip_id}: {error}") from error
            seen_ids.add(row["audio_id"])
            clips.append(row)
    if len(clips) < 2:
        raise ValueError("At least two clips are needed for an association analysis")
    return clips


def summarize_phonemes(clips):
    wers = [clip["wer"] for clip in clips]
    results = []
    for phone in PHONEMES:
        proportions = [clip[f"{phone}_proportion"] for clip in clips]
        with_sound = [clip["wer"] for clip in clips if clip[f"{phone}_count"] > 0]
        without_sound = [clip["wer"] for clip in clips if clip[f"{phone}_count"] == 0]
        mean_with = mean(with_sound) if with_sound else None
        mean_without = mean(without_sound) if without_sound else None
        coefficient = None
        if len(set(wers)) < 2:
            status = "constant_wer"
        elif len(set(proportions)) < 2:
            status = "constant_proportion"
        else:
            coefficient = correlation(proportions, wers)
            status = "exploratory" if len(with_sound) >= MIN_CLIPS_WITH_SOUND else "low_support"
        results.append({
            "phoneme": phone,
            "clips_with_phoneme": len(with_sound),
            "clips_without_phoneme": len(without_sound),
            "total_occurrences": int(sum(clip[f"{phone}_count"] for clip in clips)),
            "mean_clip_wer_with": mean_with,
            "mean_clip_wer_without": mean_without,
            "wer_difference": mean_with - mean_without
                if mean_with is not None and mean_without is not None else None,
            "proportion_wer_correlation": coefficient,
            "status": status,
        })
    return sorted(
        results,
        key=lambda row: (row["status"] == "exploratory", abs(row["proportion_wer_correlation"] or 0)),
        reverse=True,
    )


def analyze_features(input_file, output_file):
    output_file = Path(output_file)
    if output_file.exists():
        raise FileExistsError(f"Output already exists: {output_file}. Choose another --output.")
    clips = read_features(input_file)
    results = summarize_phonemes(clips)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("x", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=results[0])
        writer.writeheader()
        writer.writerows(results)
    return clips, results


def main():
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input", nargs="?", type=Path,
        default=project_dir / "outputs" / "dev-other_base.en_phonemes_complete_features.csv",
        help="CSV generated by build_features.py",
    )
    parser.add_argument("--output", type=Path, help="new output CSV (existing files are refused)")
    args = parser.parse_args()
    output_file = args.output or project_dir / "outputs" / f"{args.input.stem}_associations.csv"
    try:
        clips, results = analyze_features(args.input, output_file)
    except (OSError, ValueError, csv.Error) as error:
        parser.exit(1, f"Error: {error}\n")

    speakers = len({clip["speaker_id"] for clip in clips})
    print(f"Clips: {len(clips)} | Speakers: {speakers}")
    print(f"Average clip WER: {mean(clip['wer'] for clip in clips):.1%}")
    print(f"Clips using G2P: {sum(clip['num_g2p_words'] > 0 for clip in clips)} | "
          f"Clips with unresolved words: {sum(clip['num_unresolved_words'] > 0 for clip in clips)}")
    print("\nStrongest proportion/WER associations (at least five clips contain the sound):")
    supported = [row for row in results if row["status"] == "exploratory"]
    if not supported:
        print("No variable sounds meet the support threshold.")
    for row in supported[:5]:
        print(f"{row['phoneme']:>2}: r={row['proportion_wer_correlation']:+.3f} | "
              f"present in {row['clips_with_phoneme']}/{len(clips)} clips")
    print("\nPositive r means higher sound proportion went with higher whole-clip WER.")
    print("All 39 comparisons are exploratory; length and word choice are not controlled.")
    if speakers == 1:
        print("Only one speaker: these patterns need checking on other speakers.")
    print(f"Saved all 39 sounds to {output_file}")


if __name__ == "__main__":
    main()
