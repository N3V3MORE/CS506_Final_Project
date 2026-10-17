"""Inspect saved word errors without loading Whisper or downloading audio.

Run: python audit_word_errors.py results.csv [--output new_word_errors.csv]
Word associations use clips whose raw CMU tokens exactly match the normalized
reference tokens. Other clips remain in the overall transcription summary.
"""

import argparse
from collections import Counter
import csv
import math
from pathlib import Path
import re

from transcribe_selection import RESULT_COLUMNS, normalize_text


WORD_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?")


def audit_results(input_file, dictionary=None):
    from jiwer import process_words

    if dictionary is None:
        import cmudict
        dictionary = cmudict.dict()
    with Path(input_file).open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source, strict=True)
        if reader.fieldnames != RESULT_COLUMNS:
            raise ValueError("Expected the five-column transcription results CSV.")
        rows = list(reader)
    if not rows:
        raise ValueError("No saved transcriptions to audit.")

    clip_rows, word_rows = [], []
    seen_ids = set()
    for row in rows:
        if None in row or any(value is None for value in row.values()):
            raise ValueError("Results contain an incomplete row.")
        clip_id = row["audio_id"]
        if not clip_id or clip_id in seen_ids or not row["speaker_id"].strip():
            raise ValueError(f"Missing or duplicate audio/speaker ID: {clip_id}")
        seen_ids.add(clip_id)
        reference = normalize_text(row["reference_text"])
        prediction = normalize_text(row["predicted_text"])
        if not reference.split():
            raise ValueError(f"{clip_id}: normalized reference is empty.")
        alignment = process_words(reference, prediction)
        saved_wer = float(row["wer"])
        if not math.isfinite(saved_wer) or saved_wer < 0 or not math.isclose(saved_wer, alignment.wer, abs_tol=1e-10):
            raise ValueError(f"{clip_id}: saved WER does not match the current normalizer.")
        raw_words = WORD_PATTERN.findall(row["reference_text"].lower())
        normalized_words = reference.split()
        eligible = raw_words == normalized_words
        clip_rows.append({
            "audio_id": clip_id, "speaker_id": row["speaker_id"], "wer": saved_wer,
            "raw_num_words": len(raw_words), "normalized_num_words": len(normalized_words),
            "raw_cmudict_misses": sum(word not in dictionary for word in raw_words),
            "word_audit_eligible": eligible, "substitutions": alignment.substitutions,
            "deletions": alignment.deletions, "insertions": alignment.insertions,
            "reference_text": row["reference_text"], "predicted_text": row["predicted_text"],
        })
        if not eligible:
            continue
        for chunk in alignment.alignments[0]:
            if chunk.type == "insert":
                continue
            predicted_words = " ".join(alignment.hypotheses[0][chunk.hyp_start_idx:chunk.hyp_end_idx])
            reference_span = " ".join(raw_words[chunk.ref_start_idx:chunk.ref_end_idx])
            equal_span_lengths = chunk.ref_end_idx - chunk.ref_start_idx == chunk.hyp_end_idx - chunk.hyp_start_idx
            for index in range(chunk.ref_start_idx, chunk.ref_end_idx):
                if equal_span_lengths:
                    predicted_word = alignment.hypotheses[0][chunk.hyp_start_idx + index - chunk.ref_start_idx]
                else:
                    predicted_word = predicted_words
                word_rows.append({
                    "audio_id": clip_id, "speaker_id": row["speaker_id"], "word_index": index,
                    "reference_word": raw_words[index], "cmudict_missing": int(raw_words[index] not in dictionary),
                    "error_type": chunk.type,
                    "predicted_words": predicted_word,
                    "alignment_scope": "span" if chunk.type == "substitute" and not equal_span_lengths else "word",
                    "reference_span": reference_span, "wer": saved_wer,
                })

    totals = Counter()
    for row in clip_rows:
        for key in ("raw_num_words", "normalized_num_words", "raw_cmudict_misses",
                    "substitutions", "deletions", "insertions"):
            totals[key] += row[key]
    summary = dict(totals)
    summary.update({
        "clips": len(clip_rows), "speakers": len({row["speaker_id"] for row in clip_rows}),
        "eligible_clips": sum(row["word_audit_eligible"] for row in clip_rows),
        "zero_wer_clips": sum(row["wer"] == 0 for row in clip_rows),
        "high_wer_clips": sum(row["wer"] >= .5 for row in clip_rows),
        "wer_over_one_clips": sum(row["wer"] > 1 for row in clip_rows),
        "pooled_wer": (totals["substitutions"] + totals["deletions"] + totals["insertions"])
                       / totals["normalized_num_words"],
    })
    for label, missing in (("covered", 0), ("missing", 1)):
        words = [row for row in word_rows if row["cmudict_missing"] == missing]
        summary[f"{label}_word_count"] = len(words)
        summary[f"{label}_word_errors"] = sum(row["error_type"] != "equal" for row in words)
        summary[f"{label}_speakers"] = len({row["speaker_id"] for row in words})
    return clip_rows, word_rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, help="optional new word-error CSV")
    parser.add_argument("--examples", type=int, default=25)
    args = parser.parse_args()
    if args.examples < 0 or args.examples > 30:
        parser.error("--examples must be between 0 and 30")
    try:
        if args.output and args.output.exists():
            raise FileExistsError(f"Output already exists: {args.output}")
        clips, words, summary = audit_results(args.input)
        if args.output:
            if not words:
                raise ValueError("No clips have unchanged reference tokenization; no word CSV written.")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8", newline="") as destination:
                writer = csv.DictWriter(destination, fieldnames=words[0])
                writer.writeheader()
                writer.writerows(words)
    except (OSError, ValueError, KeyError, csv.Error, ImportError) as error:
        parser.exit(1, f"Error: {error}\n")

    print(f"Clips: {summary['clips']} | Speakers: {summary['speakers']} | Pooled WER: {summary['pooled_wer']:.2%}")
    coverage = 1 - summary["raw_cmudict_misses"] / summary["raw_num_words"] if summary["raw_num_words"] else 0
    print(f"Raw CMUdict word coverage: {coverage:.2%} | Missing occurrences: {summary['raw_cmudict_misses']}")
    print(f"Word audit: {summary['eligible_clips']} eligible clips | "
          f"{summary['clips'] - summary['eligible_clips']} excluded because normalization changed reference tokens")
    print(f"Zero WER: {summary['zero_wer_clips']} | WER >= 50%: {summary['high_wer_clips']} | "
          f"WER > 100%: {summary['wer_over_one_clips']}")
    print(f"Insertions: {summary['insertions']} (reported separately, not assigned to reference words)")
    for label in ("covered", "missing"):
        count = summary[f"{label}_word_count"]
        errors = summary[f"{label}_word_errors"]
        rate = f"{errors / count:.2%}" if count else "undefined"
        print(f"CMUdict {label}: {errors}/{count} substituted/deleted words ({rate}); "
              f"{summary[f'{label}_speakers']} speakers")
    print("\nHighest-WER clips:")
    for clip in sorted(clips, key=lambda row: (-row["wer"], row["audio_id"]))[:5]:
        print(f"{clip['audio_id']} | {clip['wer']:.2%} | {clip['reference_text']} -> {clip['predicted_text']}")
    print("\nWord-error examples (missing words first, then audio ID and position):")
    examples = [word for word in words if word["error_type"] != "equal"]
    examples.sort(key=lambda word: (-word["cmudict_missing"], word["audio_id"], word["word_index"]))
    for word in examples[:args.examples]:
        scope = f" | alignment span: {word['reference_span']}" if word["alignment_scope"] == "span" else ""
        print(f"{word['audio_id']} #{word['word_index']} | {word['reference_word']} -> "
              f"{word['predicted_words'] or '[deleted]'} | {word['error_type']} | "
              f"CMUdict missing: {bool(word['cmudict_missing'])}{scope}")
    if args.output:
        print(f"Saved word errors to {args.output}")


if __name__ == "__main__":
    main()
