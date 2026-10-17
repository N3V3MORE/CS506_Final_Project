"""CPU-only comparisons of word count, sound proportions, and CMUdict coverage.

Install: python -m pip install -r requirements-analysis.txt
Run: python evaluate_features.py [features.csv] [--output predictions.csv]

Compares the training mean, word count, and word count + phoneme proportions.
Also tests word count + the share of reference words missing from CMUdict:
(num_g2p_words + num_unresolved_words) / num_words.
Models use StandardScaler and Ridge(alpha=1.0), fitted on training clips only.
Each clip is held out once. Several speakers use up to five grouped
folds; one speaker uses five shuffled clip folds with seed 42.
Negative predictions become zero; WER above one is allowed. MSE is the primary
score, with MAE also reported. This is a preliminary evaluation on saved data.
Writes a new *_cmudict_predictions.csv under outputs/ by default.
"""

import argparse
import csv
import math
from pathlib import Path
from statistics import mean

from sklearn import __version__ as sklearn_version
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analyze_features import read_features
from build_features import PHONEMES


PHONEME_COLUMNS = [f"{phone}_proportion" for phone in PHONEMES]
PREDICTION_LABELS = (
    ("baseline_prediction", "Training-average guess"),
    ("word_count_prediction", "Word count"),
    ("word_phoneme_prediction", "Word count + phonemes"),
    ("word_cmudict_prediction", "Word count + CMUdict misses"),
)


def evaluate_clips(clips):
    if len(clips) < 5:
        raise ValueError("At least five clips are needed for the prediction check")

    speakers = [clip["speaker_id"] for clip in clips]
    num_speakers = len(set(speakers))
    if num_speakers > 1:
        splitter = GroupKFold(n_splits=min(5, num_speakers))
        folds = splitter.split(clips, groups=speakers)
        split_name = "speaker_grouped"
    else:
        splitter = KFold(n_splits=5, shuffle=True, random_state=42)
        folds = splitter.split(clips)
        split_name = "single_speaker_clips"

    features = {
        "word_count_prediction": [],
        "word_phoneme_prediction": [],
        "word_cmudict_prediction": [],
    }
    missing_proportions = []
    for clip in clips:
        try:
            num_words = float(clip["num_words"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{clip['audio_id']}: num_words must be a positive whole number") from error
        if not math.isfinite(num_words) or num_words <= 0 or not num_words.is_integer():
            raise ValueError(f"{clip['audio_id']}: num_words must be a positive whole number")
        missing_words = clip["num_g2p_words"] + clip["num_unresolved_words"]
        if missing_words > num_words:
            raise ValueError(f"{clip['audio_id']}: CMUdict-missing word count exceeds num_words")
        missing_proportion = missing_words / num_words
        missing_proportions.append(missing_proportion)
        features["word_count_prediction"].append([num_words])
        features["word_phoneme_prediction"].append(
            [num_words] + [clip[column] for column in PHONEME_COLUMNS]
        )
        features["word_cmudict_prediction"].append([num_words, missing_proportion])
    targets = [clip["wer"] for clip in clips]
    predictions = [None] * len(clips)
    for fold, (train, test) in enumerate(folds, start=1):
        training_mean = mean(targets[index] for index in train)
        for index in test:
            predictions[index] = {
                "audio_id": clips[index]["audio_id"],
                "speaker_id": speakers[index],
                "split_strategy": split_name,
                "fold": fold,
                "actual_wer": targets[index],
                "cmudict_missing_proportion": missing_proportions[index],
                "baseline_prediction": training_mean,
            }
        for column, values in features.items():
            model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            model.fit([values[index] for index in train], [targets[index] for index in train])
            estimates = model.predict([values[index] for index in test])
            for index, estimate in zip(test, estimates):
                if not math.isfinite(estimate):
                    raise ValueError(f"{clips[index]['audio_id']}: non-finite model prediction")
                predictions[index][column] = max(0.0, float(estimate))
    return predictions


def prediction_errors(predictions):
    scores = {}
    for column, _ in PREDICTION_LABELS:
        errors = [row[column] - row["actual_wer"] for row in predictions]
        scores[column] = {
            "mae": mean(abs(error) for error in errors),
            "mse": mean(error ** 2 for error in errors),
        }
    return scores


def evaluate_features(input_file, output_file):
    output_file = Path(output_file)
    if output_file.exists():
        raise FileExistsError(f"Output already exists: {output_file}. Choose another --output.")
    predictions = evaluate_clips(read_features(input_file))
    scores = prediction_errors(predictions)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("x", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=predictions[0])
        writer.writeheader()
        writer.writerows(predictions)
    return predictions, scores


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
    output_file = args.output or project_dir / "outputs" / f"{args.input.stem}_cmudict_predictions.csv"
    try:
        predictions, scores = evaluate_features(args.input, output_file)
    except (OSError, ValueError, csv.Error) as error:
        parser.exit(1, f"Error: {error}\n")

    speakers = len({row["speaker_id"] for row in predictions})
    folds = len({row["fold"] for row in predictions})
    print(f"Clips: {len(predictions)} | Speakers: {speakers} | Folds: {folds}")
    print(f"CPU | scikit-learn {sklearn_version} | Ridge alpha=1.0")
    print(f"Split: {predictions[0]['split_strategy']}")
    if speakers == 1:
        print("One speaker: this checks held-out clips from that voice. More speakers are needed.")
    print("\nHeld-out scores (lower is better; MAE is average miss in percentage points):")
    for column, label in PREDICTION_LABELS:
        score = scores[column]
        print(f"{label}: MAE={100 * score['mae']:.2f} points | MSE={score['mse']:.6f}")

    for column, reference, label in (
        ("word_count_prediction", "baseline_prediction", "Word count over mean guess"),
        ("word_phoneme_prediction", "word_count_prediction", "Adding phonemes to word count"),
        ("word_cmudict_prediction", "word_count_prediction", "Adding CMUdict misses to word count"),
    ):
        reference_mse = scores[reference]["mse"]
        if reference_mse == 0:
            print(f"{label}: MSE improvement is undefined because the reference error is zero.")
        else:
            improvement = 100 * (reference_mse - scores[column]["mse"]) / reference_mse
            print(f"{label}: {improvement:+.1f}% MSE improvement")
    print("Fixed model and folds; no tuning. These results are preliminary.")
    print(f"Saved each held-out prediction to {output_file}")


if __name__ == "__main__":
    main()
