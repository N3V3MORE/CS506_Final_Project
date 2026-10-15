"""Turn a completed pronunciation CSV into one numeric feature row per clip.

Run: python build_features.py [input.csv] [--output output.csv]
Defaults to the saved dev-other/Base.en pronunciations and writes under outputs/.
WER is copied unchanged for later analysis. Counts describe the reference
pronunciations; unresolved words still count toward the number of words.
"""

import argparse
from collections import Counter
import csv
import math
from pathlib import Path
import re


PHONEMES = (
    "AA AE AH AO AW AY B CH D DH EH ER EY F G HH "
    "IH IY JH K L M N NG OW OY P R S SH T TH UH UW V W Y Z ZH"
).split()
VOWELS = set("AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW".split())
WORD_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?")
REQUIRED_COLUMNS = {
    "audio_id", "speaker_id", "reference_text", "wer", "phonemes",
    "num_phonemes", "g2p_words", "unresolved_words",
}


def extract_features(row):
    """Calculate features without using the predicted text or WER."""
    if not row["audio_id"].strip() or not row["speaker_id"].strip():
        raise ValueError("audio_id and speaker_id must be present")

    error_rate = float(row["wer"])
    if not math.isfinite(error_rate) or error_rate < 0:
        raise ValueError("wer must be a finite, non-negative number")

    words = WORD_PATTERN.findall(row["reference_text"].lower())
    phones = row["phonemes"].split()
    if not words or not phones:
        raise ValueError("reference_text and phonemes must not be empty")
    invalid_phones = set(phones).difference(PHONEMES)
    if invalid_phones:
        raise ValueError(f"unrecognized phonemes: {', '.join(sorted(invalid_phones))}")
    if int(row["num_phonemes"]) != len(phones):
        raise ValueError("num_phonemes does not match the phoneme sequence")

    counts = Counter(phones)
    num_phonemes = len(phones)
    features = {
        "audio_id": row["audio_id"],
        "speaker_id": row["speaker_id"],
        "wer": row["wer"],
        "num_words": len(words),
        "num_phonemes": num_phonemes,
        "phonemes_per_word": num_phonemes / len(words),
        "vowel_proportion": sum(counts[phone] for phone in VOWELS) / num_phonemes,
        "num_g2p_words": len(WORD_PATTERN.findall(row["g2p_words"].lower())),
        "num_unresolved_words": len(WORD_PATTERN.findall(row["unresolved_words"].lower())),
    }
    for phone in PHONEMES:
        features[f"{phone}_count"] = counts[phone]
        features[f"{phone}_proportion"] = counts[phone] / num_phonemes
    return features


def export_features(input_file, output_file):
    input_file = Path(input_file)
    output_file = Path(output_file)
    if output_file.exists():
        raise FileExistsError(f"Output already exists: {output_file}. Choose another --output.")

    feature_rows = []
    seen_ids = set()
    with input_file.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source, strict=True)
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("CSV contains duplicate column names")

        for row_number, row in enumerate(reader, start=2):
            clip_id = row.get("audio_id") or f"CSV row {row_number}"
            try:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError("wrong number of CSV fields")
                if row["audio_id"] in seen_ids:
                    raise ValueError("duplicate audio_id")
                feature_rows.append(extract_features(row))
            except ValueError as error:
                raise ValueError(f"{clip_id}: {error}") from error
            seen_ids.add(row["audio_id"])

    if not feature_rows:
        raise ValueError("Input CSV contains no clips")

    # Validate every row before creating the output.
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("x", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=feature_rows[0])
        writer.writeheader()
        writer.writerows(feature_rows)
    return len(feature_rows)


def main():
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input", nargs="?", type=Path,
        default=project_dir / "dev-other_base.en_phonemes_complete.csv",
        help="completed pronunciation CSV",
    )
    parser.add_argument("--output", type=Path, help="new output CSV (existing files are refused)")
    args = parser.parse_args()
    output_file = args.output or project_dir / "outputs" / f"{args.input.stem}_features.csv"

    try:
        num_clips = export_features(args.input, output_file)
    except (OSError, ValueError, csv.Error) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"Saved features for {num_clips} clips to {output_file}")


if __name__ == "__main__":
    main()
