"""CPU-only WER estimates from recognized text and Whisper confidence.

Run: python evaluate_confidence.py confidence1.csv confidence2.csv --output predictions.csv
Inputs come from transcribe_selection.py --confidence and keep its metadata.
Five speaker-separated folds compare the training mean/median, recognized-text
controls, and those controls plus avg_logprob. Ridge alpha is fixed at one.
MAE in WER percentage points is primary; MSE is also shown. These saved clips
have already been studied, so this is exploratory rather than confirmation.
"""

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
from statistics import mean, median

from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from build_features import WORD_PATTERN
from transcribe_selection import CONFIDENCE_METHOD as CONFIDENCE, NORMALIZATION, RESULT_COLUMNS, TRANSCRIPTION_SETTINGS


PREDICTION_LABELS = (
    ("mean_prediction", "Training average"),
    ("median_prediction", "Training median"),
    ("text_prediction", "Recognized text"),
    ("confidence_prediction", "Recognized text + avg_logprob"),
)
TEXT_FEATURES = (
    "recognized word count", "CMUdict-missing share", "mean word length",
    "training-word rarity", "inverse recognized word count",
)


def read_confidence(paths, include_targets=True):
    rows, seen = [], set()
    for path in map(Path, paths):
        metadata = json.loads(path.with_suffix(".metadata.json").read_text(encoding="utf-8-sig"))
        if (metadata.get("model_name") != "base.en" or metadata.get("device") != "cuda:0"
                or metadata.get("effective_device") != "cuda:0"):
            raise ValueError(f"{path}: verified Base.en CUDA metadata is required")
        if (metadata.get("confidence") != CONFIDENCE or metadata.get("normalization") != NORMALIZATION
                or metadata.get("settings") != TRANSCRIPTION_SETTINGS):
            raise ValueError(f"{path}: confidence, normalization or decoding metadata differs")
        with path.open(encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source, strict=True)
            if reader.fieldnames != RESULT_COLUMNS + ["avg_logprob"]:
                raise ValueError(f"{path}: expected transcription columns plus avg_logprob")
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(f"{path}: incomplete CSV row")
                if not row["audio_id"].strip() or row["audio_id"] in seen or not row["speaker_id"].strip():
                    raise ValueError(f"{path}: missing or duplicate audio/speaker ID")
                if include_targets:
                    error_rate = float(row["wer"])
                    if not math.isfinite(error_rate) or error_rate < 0:
                        raise ValueError(f"{row['audio_id']}: WER must be finite and non-negative")
                    row["wer"] = error_rate
                confidence = float(row["avg_logprob"]) if row["avg_logprob"].strip() else None
                if confidence is not None and (not math.isfinite(confidence) or confidence > 0):
                    raise ValueError(f"{row['audio_id']}: avg_logprob must be finite and at most zero")
                row["avg_logprob"] = confidence
                if not include_targets:
                    row = {key: row[key] for key in ("audio_id", "speaker_id", "predicted_text", "avg_logprob")}
                seen.add(row["audio_id"])
                rows.append(row)
    if not rows:
        raise ValueError("No confidence rows found")
    return rows


def learn_feature_state(rows, train):
    frequencies = Counter(word for index in train
                          for word in WORD_PATTERN.findall(rows[index]["predicted_text"].lower()))
    scores = [rows[index]["avg_logprob"] for index in train if rows[index]["avg_logprob"] is not None]
    return {"word_counts": dict(frequencies), "confidence_fill": median(scores) if scores else 0.0}


def transform_features(rows, state, dictionary):
    """Use hypotheses with frozen word frequencies and confidence fill."""
    frequencies = state["word_counts"]
    denominator = sum(frequencies.values()) + len(frequencies) + 1
    values = []
    for row in rows:
        recognized = WORD_PATTERN.findall(row["predicted_text"].lower())
        count = len(recognized)
        values.append([
            count,
            sum(word not in dictionary for word in recognized) / count if count else 0.0,
            sum(map(len, recognized)) / count if count else 0.0,
            sum(-math.log((frequencies.get(word, 0) + 1) / denominator) for word in recognized) / count if count else 0.0,
            1.0 / max(1, count),
            row["avg_logprob"] if row["avg_logprob"] is not None else state["confidence_fill"],
        ])
    return values


def feature_matrix(rows, train, dictionary):
    return transform_features(rows, learn_feature_state(rows, train), dictionary)


def evaluate_clips(rows, dictionary):
    speakers = [row["speaker_id"] for row in rows]
    if len(set(speakers)) < 5:
        raise ValueError("At least five speakers are required")
    targets = [row["wer"] for row in rows]
    predictions = [None] * len(rows)
    for fold, (train, test) in enumerate(GroupKFold(n_splits=5).split(rows, groups=speakers), 1):
        training_targets = [targets[index] for index in train]
        training_mean, training_median = mean(training_targets), median(training_targets)
        for index in test:
            predictions[index] = {
                "audio_id": rows[index]["audio_id"], "speaker_id": speakers[index],
                "fold": fold, "actual_wer": targets[index],
                "mean_prediction": training_mean,
                "median_prediction": training_median,
            }
        values = feature_matrix(rows, train, dictionary)
        for column, width in (("text_prediction", 5), ("confidence_prediction", 6)):
            model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            model.fit([values[index][:width] for index in train], training_targets)
            estimates = model.predict([values[index][:width] for index in test])
            for index, estimate in zip(test, estimates):
                if not math.isfinite(estimate):
                    raise ValueError(f"{rows[index]['audio_id']}: non-finite WER prediction")
                predictions[index][column] = max(0.0, float(estimate))
    return predictions


def prediction_errors(rows):
    scores = {}
    for column, _ in PREDICTION_LABELS:
        errors = [row[column] - row["actual_wer"] for row in rows]
        scores[column] = {
            "mae": mean(abs(error) for error in errors),
            "mse": mean(error ** 2 for error in errors),
        }
    return scores


def evaluate_confidence(input_files, output_file, dictionary=None):
    output_file = Path(output_file)
    if output_file.exists():
        raise FileExistsError(f"Output already exists: {output_file}")
    rows = read_confidence(input_files)
    if dictionary is None:
        import cmudict
        dictionary = cmudict.dict()
    predictions = evaluate_clips(rows, dictionary)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("x", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=predictions[0])
        writer.writeheader()
        writer.writerows(predictions)
    return rows, predictions, prediction_errors(predictions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Base.en confidence CSVs")
    parser.add_argument("--output", required=True, type=Path, help="new predictions CSV")
    args = parser.parse_args()
    try:
        rows, predictions, scores = evaluate_confidence(args.inputs, args.output)
    except (OSError, ValueError, TypeError, KeyError, csv.Error, ImportError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"CPU | Clips: {len(rows)} | Speakers: {len({row['speaker_id'] for row in rows})} | Speaker folds: 5")
    print("Recognized-text controls: " + ", ".join(TEXT_FEATURES))
    print("Additional confidence: " + CONFIDENCE)
    print(f"Missing confidence: {sum(row['avg_logprob'] is None for row in rows)} (training-median fill; zero if all training scores are missing)")
    print("Fixed StandardScaler + Ridge(alpha=1). Negative estimates become zero; WER above one is kept.")
    print("\nOverall scores (MAE is average WER prediction miss in percentage points):")
    for column, label in PREDICTION_LABELS:
        score = scores[column]
        print(f"{label}: MAE={100 * score['mae']:.2f} points | MSE={score['mse']:.6f}")
    print("\nScores by speaker fold (MAE points / MSE):")
    for fold in range(1, 6):
        scores = prediction_errors([row for row in predictions if row["fold"] == fold])
        shown = " | ".join(f"{label}: {100 * scores[column]['mae']:.2f} / {scores[column]['mse']:.6f}"
                           for column, label in PREDICTION_LABELS)
        print(f"Fold {fold}: {shown}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
