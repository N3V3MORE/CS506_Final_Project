"""Freeze a Base confidence predictor, then check it on untouched speakers.

Freeze: python confirm_confidence.py freeze old1.csv old2.csv --history history.csv
        --selection new_selection.csv --output frozen.json
Evaluate: python confirm_confidence.py evaluate frozen.json new_confidence.csv
          --output predictions.csv
All regression and scoring is CPU-only. Frozen predictors use no reference text.
"""

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
from statistics import mean, median
import subprocess

import numpy as np
import sklearn
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluate_confidence import (
    PREDICTION_LABELS, learn_feature_state, prediction_errors,
    read_confidence, transform_features,
)
from transcribe_selection import read_selection


PROTOCOL = {
    "clips": 1000, "speakers": 40, "clips_per_speaker": 25, "seed": 507,
    "bootstrap_draws": 10000, "primary_metric": "MAE in WER percentage points",
    "success": "Confidence MAE below text MAE and paired 95% interval lower bound above zero",
}
SOURCE_FILES = ("confirm_confidence.py", "evaluate_confidence.py", "build_features.py", "transcribe_selection.py")


def file_record(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def source_hashes():
    return {name: file_record(Path(__file__).parent / name)["sha256"] for name in SOURCE_FILES}


def context(dictionary):
    return {
        "python": platform.python_version(), "numpy": np.__version__,
        "scikit-learn": sklearn.__version__, "cmudict": version("cmudict"),
        "dictionary_words_sha256": hashlib.sha256("\n".join(sorted(dictionary)).encode()).hexdigest(),
    }


def digest(frozen):
    content = {key: value for key, value in frozen.items() if key != "sha256"}
    return hashlib.sha256(json.dumps(content, sort_keys=True, allow_nan=False).encode()).hexdigest()


def check_cohort(rows, excluded_speakers, excluded_ids, protocol):
    counts = Counter(row["speaker_id"] for row in rows)
    if (len(rows) != protocol["clips"] or len(counts) != protocol["speakers"]
            or set(counts.values()) != {protocol["clips_per_speaker"]}):
        raise ValueError("New selection must contain 1,000 clips from 40 speakers, 25 each")
    if set(counts).intersection(excluded_speakers) or {row["audio_id"] for row in rows}.intersection(excluded_ids):
        raise ValueError("New selection overlaps historical speakers or audio IDs")


def freeze(input_files, history_file, selection_file, output_file, dictionary):
    output_file = Path(output_file)
    if output_file.exists():
        raise FileExistsError(f"Output already exists: {output_file}")
    paths = [Path(path) for path in input_files]
    files = [file_record(path) for path in paths + [path.with_suffix(".metadata.json") for path in paths]
             + [Path(history_file), Path(selection_file)]]
    rows = read_confidence(input_files)
    with Path(history_file).open(encoding="utf-8-sig", newline="") as source:
        history = list(csv.DictReader(source, strict=True))
    if not history or any(not row.get("speaker_id") or not row.get("audio_id") for row in history):
        raise ValueError("History must contain audio_id and speaker_id")
    excluded_speakers = sorted({row["speaker_id"] for row in rows + history})
    excluded_ids = sorted({row["audio_id"] for row in rows + history})
    selection = read_selection(selection_file)
    check_cohort(selection, excluded_speakers, excluded_ids, PROTOCOL)
    asr_versions = [json.loads(path.with_suffix(".metadata.json").read_text())["versions"] for path in paths]
    if any(item != asr_versions[0] for item in asr_versions[1:]):
        raise ValueError("Training ASR runtime versions differ")
    state = learn_feature_state(rows, range(len(rows)))
    values = transform_features(rows, state, dictionary)
    targets = [row["wer"] for row in rows]
    models = {}
    for name, width in (("text_prediction", 5), ("confidence_prediction", 6)):
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit([row[:width] for row in values], targets)
        scaler, ridge = model.named_steps["standardscaler"], model.named_steps["ridge"]
        models[name] = {
            "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
            "coef": ridge.coef_.tolist(), "intercept": float(ridge.intercept_),
        }
    frozen = {
        "version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, text=True).strip(),
        "source_hashes": source_hashes(), "context": context(dictionary),
        "inputs": files, "selection": file_record(selection_file), "asr_versions": asr_versions[0],
        "protocol": dict(PROTOCOL), "feature_state": state, "models": models,
        "mean_prediction": mean(targets), "median_prediction": median(targets),
        "excluded_speakers": excluded_speakers, "excluded_ids": excluded_ids,
        "training_clips": len(rows), "training_speakers": len({row["speaker_id"] for row in rows}),
    }
    frozen["sha256"] = digest(frozen)
    for record in files:
        if file_record(record["path"]) != record:
            raise ValueError(f"Input changed while fitting: {record['path']}")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("x", encoding="utf-8") as destination:
        json.dump(frozen, destination, indent=2, sort_keys=True, allow_nan=False)
        destination.write("\n")
    return frozen


def load_frozen(path, dictionary):
    frozen = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if frozen["version"] != 1 or frozen["sha256"] != digest(frozen):
        raise ValueError("Frozen predictor has changed")
    if frozen["source_hashes"] != source_hashes() or frozen["context"] != context(dictionary):
        raise ValueError("Source, dictionary or analysis runtime differs from the frozen predictor")
    for record in frozen["inputs"]:
        if file_record(record["path"]) != record:
            raise ValueError(f"Frozen input changed: {record['path']}")
    return frozen


def predict_frozen(rows, frozen, dictionary):
    values = np.asarray(transform_features(rows, frozen["feature_state"], dictionary))
    predictions = [
        {"audio_id": row["audio_id"], "speaker_id": row["speaker_id"],
         "mean_prediction": frozen["mean_prediction"], "median_prediction": frozen["median_prediction"]}
        for row in rows
    ]
    for name, state in frozen["models"].items():
        width = len(state["coef"])
        estimates = ((values[:, :width] - state["mean"]) / state["scale"]) @ state["coef"] + state["intercept"]
        if not np.isfinite(estimates).all():
            raise ValueError("Frozen predictor returned a non-finite estimate")
        for row, value in zip(predictions, estimates):
            row[name] = max(0.0, float(value))
    return predictions


def paired_interval(rows, protocol):
    differences = defaultdict(list)
    for row in rows:
        differences[row["speaker_id"]].append(
            abs(row["text_prediction"] - row["actual_wer"]) - abs(row["confidence_prediction"] - row["actual_wer"])
        )
    values = np.array([mean(differences[speaker]) for speaker in sorted(differences)])
    samples = np.random.default_rng(protocol["seed"]).integers(0, len(values), size=(protocol["bootstrap_draws"], len(values)))
    return np.quantile(100 * values[samples].mean(axis=1), [0.025, 0.975]).tolist()


def evaluate(frozen_file, input_files, output_file, dictionary):
    output_file = Path(output_file)
    if output_file.exists():
        raise FileExistsError(f"Output already exists: {output_file}")
    frozen = load_frozen(frozen_file, dictionary)
    selection = {row["audio_id"]: row for row in read_selection(frozen["selection"]["path"])}
    records = [file_record(path) for path in input_files]
    rows = read_confidence(input_files, include_targets=False)
    check_cohort(rows, frozen["excluded_speakers"], frozen["excluded_ids"], frozen["protocol"])
    if {row["audio_id"] for row in rows} != set(selection) or any(row["speaker_id"] != selection[row["audio_id"]]["speaker_id"] for row in rows):
        raise ValueError("Confidence rows do not match the frozen selection")
    for path in map(Path, input_files):
        metadata = json.loads(path.with_suffix(".metadata.json").read_text())
        if metadata["selection_sha256"] != frozen["selection"]["sha256"] or metadata["versions"] != frozen["asr_versions"]:
            raise ValueError("New ASR selection or runtime differs from the frozen study")
    predictions = predict_frozen(rows, frozen, dictionary)
    # References and numerical WERs enter only after all predictions are fixed.
    targets = read_confidence(input_files)
    for prediction, target in zip(predictions, targets):
        if prediction["audio_id"] != target["audio_id"]:
            raise ValueError("New confidence row order changed during evaluation")
        if target["reference_text"] != selection[target["audio_id"]]["reference_text"]:
            raise ValueError("New reference differs from the frozen selection")
        prediction["actual_wer"] = target["wer"]
    if any(file_record(record["path"]) != record for record in records):
        raise ValueError("New confidence input changed during evaluation")
    scores = prediction_errors(predictions)
    interval = paired_interval(predictions, frozen["protocol"])
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("x", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=predictions[0])
        writer.writeheader()
        writer.writerows(predictions)
    return predictions, scores, interval


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("freeze")
    training.add_argument("inputs", nargs="+", type=Path)
    training.add_argument("--history", required=True, type=Path)
    training.add_argument("--selection", required=True, type=Path)
    testing = commands.add_parser("evaluate")
    testing.add_argument("frozen", type=Path)
    testing.add_argument("inputs", nargs="+", type=Path)
    for command in (training, testing):
        command.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        import cmudict
        dictionary = cmudict.dict()
        if args.command == "freeze":
            frozen = freeze(args.inputs, args.history, args.selection, args.output, dictionary)
            print(f"Froze {frozen['training_clips']} training clips / {frozen['training_speakers']} speakers before new inference")
        else:
            rows, scores, interval = evaluate(args.frozen, args.inputs, args.output, dictionary)
            print(f"Held out: {len(rows)} clips / {len({row['speaker_id'] for row in rows})} speakers")
            for column, label in PREDICTION_LABELS:
                print(f"{label}: MAE={100 * scores[column]['mae']:.2f} points | MSE={scores[column]['mse']:.6f}")
            for column, label in PREDICTION_LABELS[:-1]:
                reference = scores[column]["mae"]
                gain = 100 * (reference - scores["confidence_prediction"]["mae"]) / reference if reference else None
                print(f"Confidence MAE improvement over {label}: {gain}%")
            print(f"Paired text-minus-confidence MAE reduction, 95% interval: {interval} WER points")
            print(f"Primary improvement supported: {scores['confidence_prediction']['mae'] < scores['text_prediction']['mae'] and interval[0] > 0}")
        print(f"Saved to {args.output}")
    except (OSError, ValueError, KeyError, TypeError, csv.Error, ImportError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
