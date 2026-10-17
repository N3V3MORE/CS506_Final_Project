"""CPU-only, speaker-separated checks of lexical and phonetic WER predictors.

Develop: python study_evaluation.py develop features.jsonl --results base.en=results.csv
         --folds old_predictions.csv --output new_directory --allow-historical-base
Freeze: use all three model=results.csv inputs in the development command.
Confirm: python study_evaluation.py confirm frozen.json fresh_features.jsonl
         --results tiny.en=tiny.csv base.en=base.csv small.en=small.csv --output new_directory

Confirmation applies saved development fits before reading fresh WERs. It does
not select features, tune models, or fit anything to the confirmation speakers.
"""

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess

import numpy as np
import sklearn
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from build_features import PHONEMES, WORD_PATTERN


MODELS = ("tiny.en", "base.en", "small.en")
BLOCKS = ("proportions", "bigrams", "stress")
CONTROLS = (
    "num_words", "cmudict_missing_proportion", "mean_word_length",
    "training_word_rarity", "inverse_normalized_num_words",
)
SPEC = {
    "version": 1,
    "controls": list(CONTROLS),
    "phonemes": PHONEMES,
    "blocks": list(BLOCKS),
    "alphas": [1.0, 10.0, 100.0],
    "inner_folds": 3,
    "outer_folds": 5,
    "bigram_min_count": 50,
    "bigram_min_speakers": 5,
    "bigram_max_features": 50,
    "bootstrap_draws": 10000,
    "bootstrap_seed": 506,
    "practical_gain_percent": 5.0,
    "confirmation_clips": 2000,
    "confirmation_speakers": 80,
    "confirmation_clips_per_speaker": 25,
}
CODE_FILES = (
    "study_evaluation.py", "study_features.py", "build_features.py",
    "transcribe_selection.py", "select_clips.py",
)
RESULT_COLUMNS = ["audio_id", "speaker_id", "reference_text", "predicted_text", "wer"]


def file_record(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def check_record(record):
    if file_record(record["path"]) != record:
        raise ValueError(f"Frozen input changed: {record['path']}")


def finite_number(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be finite and at least {minimum}")
    return float(value)


def whole_number(value, name, minimum=0):
    number = finite_number(value, name, minimum)
    if not number.is_integer():
        raise ValueError(f"{name} must be a whole number")
    return int(number)


def read_features(path):
    clips = []
    seen = set()
    with Path(path).open(encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                raise ValueError(f"Feature line {line_number} is empty")
            clip = json.loads(line)
            if not isinstance(clip, dict):
                raise ValueError(f"Feature line {line_number} must be an object")
            if {"wer", "actual_wer", "predicted_text"}.intersection(clip):
                raise ValueError("The feature cache must not contain WERs or predictions")
            for key in ("audio_id", "speaker_id", "chapter_id", "reference_text"):
                if not isinstance(clip.get(key), str) or not clip[key].strip():
                    raise ValueError(f"Feature line {line_number}: missing {key}")
            if clip["audio_id"] in seen:
                raise ValueError(f"Duplicate feature ID: {clip['audio_id']}")
            words = WORD_PATTERN.findall(clip["reference_text"].lower())
            if not words or clip["words"] != words:
                raise ValueError(f"{clip['audio_id']}: words do not match the reference tokenizer")
            if whole_number(clip["num_words"], "num_words", 1) != len(words):
                raise ValueError(f"{clip['audio_id']}: num_words does not match words")
            whole_number(clip["normalized_num_words"], "normalized_num_words", 1)
            finite_number(clip["mean_word_length"], "mean_word_length")
            if not math.isclose(clip["mean_word_length"], sum(map(len, words)) / len(words)):
                raise ValueError(f"{clip['audio_id']}: mean_word_length does not match words")
            for key in ("cmudict_missing_proportion", "long_word_share", "noninitial_stress_share"):
                if finite_number(clip[key], key) > 1:
                    raise ValueError(f"{clip['audio_id']}: {key} exceeds one")
            proportions = clip["phoneme_proportions"]
            if not isinstance(proportions, list) or len(proportions) != len(PHONEMES):
                raise ValueError(f"{clip['audio_id']}: expected 39 phoneme proportions")
            for value in proportions:
                if finite_number(value, "phoneme proportion") > 1:
                    raise ValueError("Phoneme proportions must not exceed one")
            num_phonemes = whole_number(clip["num_phonemes"], "num_phonemes")
            expected_total = 1.0 if num_phonemes else 0.0
            if not math.isclose(sum(proportions), expected_total, abs_tol=1e-8):
                raise ValueError(f"{clip['audio_id']}: phoneme proportions have the wrong total")
            bigrams = clip["bigram_counts"]
            if not isinstance(bigrams, dict):
                raise ValueError("bigram_counts must be an object")
            for pair, count in bigrams.items():
                phones = pair.split("|")
                if len(phones) != 2 or any(phone not in PHONEMES for phone in phones):
                    raise ValueError(f"Invalid bigram: {pair}")
                whole_number(count, "bigram count", 1)
            if whole_number(clip["num_pairs"], "num_pairs") != sum(bigrams.values()):
                raise ValueError(f"{clip['audio_id']}: num_pairs does not match bigram counts")
            seen.add(clip["audio_id"])
            clips.append(clip)
    if not clips:
        raise ValueError("Feature cache contains no clips")
    return clips


def read_folds(path, clips):
    expected = {clip["audio_id"]: clip for clip in clips}
    folds = {}
    speaker_folds = defaultdict(set)
    chapter_folds = defaultdict(set)
    with Path(path).open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source, strict=True)
        if not {"audio_id", "speaker_id", "fold"}.issubset(reader.fieldnames or []):
            raise ValueError("Fold CSV needs audio_id, speaker_id, and fold")
        for row in reader:
            clip_id = row["audio_id"]
            if None in row or any(value is None for value in row.values()):
                raise ValueError("Fold CSV contains an incomplete row")
            if clip_id not in expected or clip_id in folds:
                raise ValueError(f"Unexpected or duplicate fold ID: {clip_id}")
            if row["speaker_id"] != expected[clip_id]["speaker_id"]:
                raise ValueError(f"{clip_id}: fold speaker differs from feature speaker")
            fold = int(row["fold"])
            if fold not in range(1, 6):
                raise ValueError("Development folds must be numbered 1 through 5")
            folds[clip_id] = fold
            speaker_folds[row["speaker_id"]].add(fold)
            chapter_folds[expected[clip_id]["chapter_id"]].add(fold)
    if set(folds) != set(expected) or set(folds.values()) != set(range(1, 6)):
        raise ValueError("Fold CSV must cover every feature ID in exactly five folds")
    if any(len(values) != 1 for values in speaker_folds.values()):
        raise ValueError("A speaker appears in multiple development folds")
    if any(len(values) != 1 for values in chapter_folds.values()):
        raise ValueError("A chapter appears in multiple development folds")
    return folds


def read_results(path, clips, model, allow_historical=False):
    path = Path(path)
    by_id = {clip["audio_id"]: clip for clip in clips}
    targets = {}
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source, strict=True)
        if reader.fieldnames != RESULT_COLUMNS:
            raise ValueError(f"{path}: expected the five-column transcription CSV")
        for row in reader:
            clip_id = row["audio_id"]
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{path}: incomplete result row")
            if clip_id not in by_id or clip_id in targets:
                raise ValueError(f"{path}: unexpected or duplicate ID {clip_id}")
            if any(row[key] != by_id[clip_id][key] for key in ("speaker_id", "reference_text")):
                raise ValueError(f"{clip_id}: results do not match features")
            value = float(row["wer"])
            targets[clip_id] = finite_number(value, "WER")
    if set(targets) != set(by_id):
        raise ValueError(f"{path}: results must contain every feature ID")
    provenance = {"results": file_record(path)}
    metadata_path = path.with_suffix(".metadata.json")
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
        if metadata.get("model_name") != model:
            raise ValueError(f"{metadata_path}: model does not match {model}")
        if metadata.get("device") != "cuda:0" or metadata.get("effective_device") != "cuda:0":
            raise ValueError(f"{metadata_path}: verified CUDA device is required")
        from transcribe_selection import NORMALIZATION, TRANSCRIPTION_SETTINGS

        if metadata.get("settings") != TRANSCRIPTION_SETTINGS:
            raise ValueError(f"{metadata_path}: transcription settings do not match")
        if metadata.get("normalization") != NORMALIZATION:
            raise ValueError(f"{metadata_path}: WER normalization does not match")
        if not isinstance(metadata.get("selection_sha256"), str) or len(metadata["selection_sha256"]) != 64:
            raise ValueError(f"{metadata_path}: missing selection hash")
        versions = metadata.get("versions", {})
        for key in ("python", "openai-whisper", "torch", "jiwer"):
            value = versions.get(key)
            if not isinstance(value, str) or not value.strip() or value.lower() in ("unavailable", "unknown", "none"):
                raise ValueError(f"{metadata_path}: missing actual {key} runtime version")
        provenance["metadata"] = file_record(metadata_path)
        provenance["selection_sha256"] = metadata["selection_sha256"]
        provenance["runtime_versions"] = {key: versions[key] for key in ("python", "openai-whisper", "torch", "jiwer")}
        provenance["normalization"] = metadata["normalization"]
    elif allow_historical and model == "base.en":
        provenance["historical_gpu_audit"] = "Existing Base.en development run; no metadata sidecar"
    else:
        raise ValueError(f"Missing GPU provenance sidecar: {metadata_path}")
    return [targets[clip["audio_id"]] for clip in clips], provenance


def read_history(path, clips):
    speakers = set()
    audio_ids = set()
    with Path(path).open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source, strict=True)
        if reader.fieldnames != ["dataset", "audio_id", "speaker_id"]:
            raise ValueError("History CSV needs dataset, audio_id, speaker_id")
        for row in reader:
            if None in row or any(value is None or not value.strip() for value in row.values()):
                raise ValueError("History CSV contains an incomplete row")
            speakers.add(row["speaker_id"])
            audio_ids.add(row["audio_id"])
    if not {clip["audio_id"] for clip in clips}.issubset(audio_ids):
        raise ValueError("History CSV must include every development audio ID")
    if not {clip["speaker_id"] for clip in clips}.issubset(speakers):
        raise ValueError("History CSV must include every development speaker")
    return sorted(speakers), sorted(audio_ids)


def learn_features(clips, block="lexical"):
    if block not in ("lexical",) + BLOCKS:
        raise ValueError(f"Unknown feature block: {block}")
    words = Counter(word for clip in clips for word in clip["words"])
    state = {
        "block": block, "word_counts": dict(words), "word_total": sum(words.values()),
        "feature_names": list(CONTROLS), "bigrams": [],
    }
    if block == "proportions":
        state["feature_names"] += [f"{phone}_proportion" for phone in PHONEMES]
    elif block == "bigrams":
        counts = Counter()
        speakers = defaultdict(set)
        for clip in clips:
            counts.update(clip["bigram_counts"])
            for pair in clip["bigram_counts"]:
                speakers[pair].add(clip["speaker_id"])
        vocabulary = [
            pair for pair, count in counts.items()
            if count >= SPEC["bigram_min_count"]
            and len(speakers[pair]) >= SPEC["bigram_min_speakers"]
        ]
        state["bigrams"] = sorted(vocabulary, key=lambda pair: (-counts[pair], pair))[:SPEC["bigram_max_features"]]
        if not state["bigrams"]:
            return None
        state["feature_names"] += [f"bigram:{pair}" for pair in state["bigrams"]]
    elif block == "stress":
        state["feature_names"] += ["long_word_share", "noninitial_stress_share"]
    return state


def feature_matrix(clips, state):
    counts = state["word_counts"]
    denominator = state["word_total"] + len(counts) + 1
    values = []
    for clip in clips:
        rarity = sum(-math.log((counts.get(word, 0) + 1) / denominator) for word in clip["words"])
        row = [
            clip["num_words"], clip["cmudict_missing_proportion"], clip["mean_word_length"],
            rarity / len(clip["words"]), 1.0 / clip["normalized_num_words"],
        ]
        if state["block"] == "proportions":
            row += clip["phoneme_proportions"]
        elif state["block"] == "bigrams":
            pairs = clip["num_pairs"]
            row += [clip["bigram_counts"].get(pair, 0) / pairs if pairs else 0 for pair in state["bigrams"]]
        elif state["block"] == "stress":
            row += [clip["long_word_share"], clip["noninitial_stress_share"]]
        values.append(row)
    matrix = np.asarray(values, dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError("Feature matrix contains non-finite values")
    return matrix


def fit_model(clips, targets, block, alpha):
    state = learn_features(clips, block)
    if state is None:
        return None
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(feature_matrix(clips, state), targets)
    scaler = model.named_steps["standardscaler"]
    ridge = model.named_steps["ridge"]
    state.update({
        "alpha": float(alpha), "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(), "coefficients": ridge.coef_.tolist(),
        "intercept": float(ridge.intercept_),
    })
    if not np.isfinite(np.asarray(state["coefficients"])).all() or not math.isfinite(state["intercept"]):
        raise ValueError("Fitted model contains non-finite coefficients")
    return state


def predict_model(clips, state):
    values = feature_matrix(clips, state)
    predictions = (values - np.asarray(state["scaler_mean"])) / np.asarray(state["scaler_scale"])
    predictions = predictions @ np.asarray(state["coefficients"]) + state["intercept"]
    if not np.isfinite(predictions).all():
        raise ValueError("Model produced non-finite predictions")
    return np.maximum(predictions, 0).tolist()


def choose_alpha(clips, targets, block, n_splits=3, fold_map=None):
    speakers = [clip["speaker_id"] for clip in clips]
    if len(set(speakers)) < n_splits:
        raise ValueError(f"Need at least {n_splits} training speakers for alpha selection")
    if fold_map is None:
        folds = list(GroupKFold(n_splits=n_splits).split(clips, groups=speakers))
    else:
        folds = [
            (
                [index for index, clip in enumerate(clips) if fold_map[clip["audio_id"]] != fold],
                [index for index, clip in enumerate(clips) if fold_map[clip["audio_id"]] == fold],
            )
            for fold in sorted(set(fold_map.values()))
        ]
    scores = {}
    for alpha in SPEC["alphas"]:
        squared_errors = []
        for train, test in folds:
            training = [clips[index] for index in train]
            state = fit_model(training, [targets[index] for index in train], block, alpha)
            if state is None:
                return None
            predictions = predict_model([clips[index] for index in test], state)
            squared_errors += [
                (estimate - targets[index]) ** 2 for index, estimate in zip(test, predictions)
            ]
        scores[alpha] = math.fsum(squared_errors) / len(squared_errors)
    return min(scores, key=lambda alpha: (scores[alpha], alpha))


def score_rows(rows, column):
    by_speaker = defaultdict(list)
    for row in rows:
        by_speaker[row["speaker_id"]].append(row[column] - row["actual_wer"])
    errors = [error for group in by_speaker.values() for error in group]
    return {
        "mse": math.fsum(error ** 2 for error in errors) / len(errors),
        "mae": math.fsum(abs(error) for error in errors) / len(errors),
        "speaker_balanced_mse": sum(sum(error ** 2 for error in group) / len(group) for group in by_speaker.values()) / len(by_speaker),
        "speaker_balanced_mae": sum(sum(abs(error) for error in group) / len(group) for group in by_speaker.values()) / len(by_speaker),
    }


def evaluate_development(clips, targets, folds, model_name):
    predictions = [
        {
            "audio_id": clip["audio_id"], "speaker_id": clip["speaker_id"],
            "chapter_id": clip["chapter_id"], "fold": folds[clip["audio_id"]],
            "actual_wer": target,
        }
        for clip, target in zip(clips, targets)
    ]
    unavailable = set()
    chosen_alphas = defaultdict(dict)
    for fold in range(1, 6):
        train = [index for index, clip in enumerate(clips) if folds[clip["audio_id"]] != fold]
        test = [index for index, clip in enumerate(clips) if folds[clip["audio_id"]] == fold]
        training = [clips[index] for index in train]
        training_targets = [targets[index] for index in train]
        print(f"{model_name}: development fold {fold}/5", flush=True)
        training_mean = sum(training_targets) / len(training_targets)
        for index in test:
            predictions[index]["mean_prediction"] = training_mean
        for block in ("lexical",) + BLOCKS:
            if block in unavailable:
                continue
            alpha = choose_alpha(training, training_targets, block)
            if alpha is None:
                unavailable.add(block)
                continue
            state = fit_model(training, training_targets, block, alpha)
            estimates = predict_model([clips[index] for index in test], state)
            chosen_alphas[block][str(fold)] = alpha
            for index, estimate in zip(test, estimates):
                predictions[index][f"{block}_prediction"] = estimate
    for block in unavailable:
        for row in predictions:
            row.pop(f"{block}_prediction", None)
    scores = {"mean": score_rows(predictions, "mean_prediction")}
    scores.update({
        block: score_rows(predictions, f"{block}_prediction")
        for block in ("lexical",) + BLOCKS if block not in unavailable
    })
    return predictions, {
        "scores": scores, "outer_alphas": dict(chosen_alphas),
        "unavailable_blocks": sorted(unavailable),
    }


def select_block(clips, summary):
    candidates = []
    for block in BLOCKS:
        if block in summary["scores"]:
            state = learn_features(clips, block)
            candidates.append((summary["scores"][block]["mse"], len(state["feature_names"]), block))
    if not candidates:
        raise ValueError("No phonetic block was available in every development fold")
    return min(candidates)[2]


def runtime_versions():
    return {"python": platform.python_version(), "numpy": np.__version__, "scikit-learn": sklearn.__version__}


def code_records():
    directory = Path(__file__).resolve().parent
    return {name: file_record(directory / name) for name in CODE_FILES}


def freeze_digest(manifest):
    content = {key: value for key, value in manifest.items() if key != "freeze_sha256"}
    return hashlib.sha256(json.dumps(content, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as destination:
        json.dump(value, destination, indent=2, sort_keys=True, allow_nan=False)
        destination.write("\n")


def write_predictions(path, rows):
    with Path(path).open("x", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def check_selection_hashes(provenance):
    hashes = {entry["selection_sha256"] for entry in provenance.values() if "selection_sha256" in entry}
    if len(hashes) > 1:
        raise ValueError("Model results were made from different selection files")


def check_result_runtime(provenance, expected=None):
    versions = [entry["runtime_versions"] for entry in provenance.values() if "runtime_versions" in entry]
    if expected is None and versions:
        expected = versions[0]
    if any(version != expected for version in versions):
        raise ValueError("ASR runtime versions differ between model runs or from the frozen run")
    return expected


def clean_git_head():
    directory = Path(__file__).resolve().parent
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=directory, text=True,
    )
    if status.strip():
        raise ValueError("Commit tracked changes before freezing the study")
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", *CODE_FILES],
        cwd=directory, capture_output=True, text=True,
    )
    if tracked.returncode:
        raise ValueError("Commit every study source file before freezing the study")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=directory, text=True).strip()


def check_inputs(inputs):
    check_record(inputs["features"])
    check_record(inputs["folds"])
    if "history" in inputs:
        check_record(inputs["history"])
    for provenance in inputs["results"].values():
        check_record(provenance["results"])
        if "metadata" in provenance:
            check_record(provenance["metadata"])


def develop(feature_file, result_files, fold_file, output_dir, allow_historical=False, history_file=None):
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    if set(result_files) not in ({"base.en"}, set(MODELS)):
        raise ValueError("Development needs Base.en alone or all three models")
    if set(result_files) == set(MODELS) and history_file is None:
        raise ValueError("A complete historical-ID CSV is required to freeze: use --history")
    features_record = file_record(feature_file)
    folds_record = file_record(fold_file)
    history_record = file_record(history_file) if history_file is not None else None
    clips = read_features(feature_file)
    history = read_history(history_file, clips) if history_file is not None else None
    folds = read_folds(fold_file, clips)
    targets = {}
    provenance = {}
    for model, path in result_files.items():
        targets[model], provenance[model] = read_results(path, clips, model, allow_historical)
    check_selection_hashes(provenance)
    transcription_versions = check_result_runtime(provenance)
    predictions = {}
    summaries = {}
    for model in MODELS:
        if model in targets:
            predictions[model], summaries[model] = evaluate_development(clips, targets[model], folds, model)
    selected = select_block(clips, summaries["base.en"])
    summary = {
        "stage": "development", "clips": len(clips),
        "speakers": len({clip["speaker_id"] for clip in clips}),
        "selected_block": selected, "selection_metric": "Base.en outer-fold clip MSE",
        "models": summaries, "spec": SPEC,
        "status": "preliminary_base_only" if len(targets) == 1 else "frozen_for_confirmation",
    }
    inputs = {"features": features_record, "folds": folds_record, "results": provenance}
    if history_record:
        inputs["history"] = history_record
    manifest = None
    if set(targets) == set(MODELS):
        states = {}
        for model in MODELS:
            states[model] = {}
            for block in ("lexical", selected):
                alpha = choose_alpha(clips, targets[model], block, n_splits=5, fold_map=folds)
                if alpha is None:
                    raise ValueError(f"{block} unavailable during full-development alpha selection")
                states[model][block] = fit_model(clips, targets[model], block, alpha)
        git_head = clean_git_head()
        manifest = {
            "format_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
            "git_head": git_head, "code": code_records(), "versions": runtime_versions(),
            "spec": SPEC, "inputs": inputs, "selected_block": selected, "models": states,
            "historical_speakers": history[0], "historical_ids": history[1],
            "transcription_versions": transcription_versions,
            "training_speakers": sorted({clip["speaker_id"] for clip in clips}),
            "training_ids": [clip["audio_id"] for clip in clips],
            "training_mean": {model: sum(values) / len(values) for model, values in targets.items()},
        }
        manifest["freeze_sha256"] = freeze_digest(manifest)
    check_inputs(inputs)
    output_dir.mkdir(parents=True, exist_ok=False)
    for model, rows in predictions.items():
        write_predictions(output_dir / f"{model}_development_predictions.csv", rows)
    write_json(output_dir / "summary.json", summary)
    if manifest:
        write_json(output_dir / "frozen.json", manifest)
    return summary


def bootstrap_summary(rows_by_model, draws=10000, seed=506):
    if set(rows_by_model) != set(MODELS):
        raise ValueError("Bootstrap needs all three ASR models")
    speaker_sets = [{row["speaker_id"] for row in rows} for rows in rows_by_model.values()]
    if any(speakers != speaker_sets[0] for speakers in speaker_sets[1:]):
        raise ValueError("Models have different confirmation speakers")
    speakers = sorted(speaker_sets[0])
    if len(speakers) < 2:
        raise ValueError("Bootstrap needs at least two speakers")
    sample = np.random.default_rng(seed).integers(0, len(speakers), size=(draws, len(speakers)))
    result = {}
    for model in MODELS:
        rows = rows_by_model[model]
        grouped = defaultdict(list)
        for row in rows:
            grouped[row["speaker_id"]].append(row)
        mse = []
        mae = []
        for speaker in speakers:
            group = grouped[speaker]
            mse.append([
                sum((row[column] - row["actual_wer"]) ** 2 for row in group) / len(group)
                for column in ("lexical_prediction", "phonetic_prediction")
            ])
            mae.append([
                sum(abs(row[column] - row["actual_wer"]) for row in group) / len(group)
                for column in ("lexical_prediction", "phonetic_prediction")
            ])
        mse = np.asarray(mse)
        mae = np.asarray(mae)
        observed = mse.mean(axis=0)
        sampled = mse[sample].mean(axis=1)
        gain = 100 * (observed[0] - observed[1]) / observed[0] if observed[0] > 0 else None
        zero_draws = int(np.count_nonzero(sampled[:, 0] == 0))
        ci95 = simultaneous = None
        if zero_draws == 0:
            gains = 100 * (sampled[:, 0] - sampled[:, 1]) / sampled[:, 0]
            ci95 = np.quantile(gains, [0.025, 0.975]).tolist()
            simultaneous = np.quantile(gains, [0.05 / 6, 1 - 0.05 / 6]).tolist()
        decision = "inconclusive"
        if ci95 is not None:
            if gain >= SPEC["practical_gain_percent"] and ci95[0] > 0:
                decision = "positive"
            elif ci95[1] < SPEC["practical_gain_percent"]:
                decision = "meaningful_gain_bounded_below_target"
        result[model] = {
            "speaker_balanced_lexical_mse": float(observed[0]),
            "speaker_balanced_phonetic_mse": float(observed[1]),
            "speaker_balanced_lexical_mae": float(mae[:, 0].mean()),
            "speaker_balanced_phonetic_mae": float(mae[:, 1].mean()),
            "relative_mse_gain_percent": gain, "ci95_percent": ci95,
            "bonferroni_ci98_333333_percent": simultaneous,
            "undefined_bootstrap_draws": zero_draws, "decision": decision,
            "small_positive_gain": bool(gain is not None and 0 < gain < 5 and ci95 is not None and ci95[0] > 0),
        }
    across_models = all(
        result[model]["relative_mse_gain_percent"] is not None
        and result[model]["relative_mse_gain_percent"] >= 5
        and result[model]["bonferroni_ci98_333333_percent"] is not None
        and result[model]["bonferroni_ci98_333333_percent"][0] > 0
        for model in MODELS
    )
    return {
        "models": result, "across_models_positive": across_models,
        "speakers": len(speakers), "bootstrap_draws": draws, "bootstrap_seed": seed,
        "resampling_unit": "whole speaker; identical paired draws across ASR models",
    }


def load_frozen(path):
    manifest = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if manifest.get("format_version") != 1 or manifest.get("freeze_sha256") != freeze_digest(manifest):
        raise ValueError("Frozen manifest is incompatible or has changed")
    if manifest["spec"] != SPEC or manifest["versions"] != runtime_versions():
        raise ValueError("Evaluation settings or analysis runtime differ from the frozen run")
    if manifest["code"] != code_records():
        raise ValueError("Frozen code differs from the currently executing source files")
    for record in manifest["code"].values():
        check_record(record)
    if "history" not in manifest["inputs"]:
        raise ValueError("Frozen study lacks the complete historical-ID file")
    check_inputs(manifest["inputs"])
    if set(manifest["models"]) != set(MODELS):
        raise ValueError("Confirmation requires a freeze from all three development models")
    if manifest["selected_block"] not in BLOCKS:
        raise ValueError("Frozen phonetic recipe is invalid")
    return manifest


def check_confirmation_cohort(clips, manifest):
    speakers = Counter(clip["speaker_id"] for clip in clips)
    if len(clips) != SPEC["confirmation_clips"] or len(speakers) != SPEC["confirmation_speakers"]:
        raise ValueError("Confirmation requires 2,000 clips from 80 speakers")
    if set(speakers.values()) != {SPEC["confirmation_clips_per_speaker"]}:
        raise ValueError("Confirmation requires exactly 25 clips per speaker")
    if set(speakers).intersection(manifest["historical_speakers"]):
        raise ValueError("Confirmation speakers overlap historical speakers")
    if {clip["audio_id"] for clip in clips}.intersection(manifest["historical_ids"]):
        raise ValueError("Confirmation audio IDs overlap historical IDs")


def confirm(frozen_file, feature_file, result_files, output_dir):
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    if set(result_files) != set(MODELS):
        raise ValueError("Confirmation needs complete Tiny.en, Base.en, and Small.en results")
    frozen_record = file_record(frozen_file)
    manifest = load_frozen(frozen_file)
    features_record = file_record(feature_file)
    clips = read_features(feature_file)
    check_confirmation_cohort(clips, manifest)
    training = read_features(manifest["inputs"]["features"]["path"])
    if {clip["chapter_id"] for clip in clips}.intersection(clip["chapter_id"] for clip in training):
        raise ValueError("Confirmation chapters overlap development chapters")
    selected = manifest["selected_block"]
    predictions = {}
    # Generate every prediction before opening any confirmation result CSV.
    for model in MODELS:
        lexical = predict_model(clips, manifest["models"][model]["lexical"])
        phonetic = predict_model(clips, manifest["models"][model][selected])
        predictions[model] = [
            {
                "audio_id": clip["audio_id"], "speaker_id": clip["speaker_id"],
                "chapter_id": clip["chapter_id"], "lexical_prediction": first,
                "phonetic_prediction": second,
                "mean_prediction": manifest["training_mean"][model],
            }
            for clip, first, second in zip(clips, lexical, phonetic)
        ]
    provenance = {}
    for model in MODELS:
        targets, provenance[model] = read_results(result_files[model], clips, model)
        for row, target in zip(predictions[model], targets):
            row["actual_wer"] = target
    check_selection_hashes(provenance)
    check_result_runtime(provenance, manifest["transcription_versions"])
    summary = bootstrap_summary(predictions, SPEC["bootstrap_draws"], SPEC["bootstrap_seed"])
    summary.update({
        "stage": "confirmation", "clips": len(clips), "selected_block": selected,
        "frozen_manifest": frozen_record, "features": features_record,
        "results": provenance, "freeze_sha256": manifest["freeze_sha256"],
        "interpretation": "Bounds apply to the selected recipe, dataset and ASR models, not all phonetic information.",
    })
    check_record(frozen_record)
    check_record(features_record)
    check_inputs(manifest["inputs"])
    for entry in provenance.values():
        check_record(entry["results"])
        check_record(entry["metadata"])
    output_dir.mkdir(parents=True, exist_ok=False)
    for model, rows in predictions.items():
        write_predictions(output_dir / f"{model}_confirmation_predictions.csv", rows)
    write_json(output_dir / "summary.json", summary)
    return summary


def parse_result_files(values):
    result = {}
    for value in values:
        model, separator, path = value.partition("=")
        if separator != "=" or model not in MODELS or not path or model in result:
            raise ValueError("Use each model once: tiny.en=path base.en=path small.en=path")
        result[model] = Path(path)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    development = commands.add_parser("develop")
    development.add_argument("features", type=Path)
    development.add_argument("--folds", type=Path, required=True)
    development.add_argument("--allow-historical-base", action="store_true")
    development.add_argument("--history", type=Path, help="all previously used dataset/audio_id/speaker_id rows; required to freeze")
    confirmation = commands.add_parser("confirm")
    confirmation.add_argument("frozen", type=Path)
    confirmation.add_argument("features", type=Path)
    for command in (development, confirmation):
        command.add_argument("--results", nargs="+", required=True, metavar="MODEL=CSV")
        command.add_argument("--output", type=Path, required=True, help="new directory; existing directories are refused")
    args = parser.parse_args()
    try:
        files = parse_result_files(args.results)
        if args.command == "develop":
            summary = develop(args.features, files, args.folds, args.output, args.allow_historical_base, args.history)
            print(f"Selected phonetic block: {summary['selected_block']} | {summary['status']}")
            for model, report in summary["models"].items():
                for recipe, score in report["scores"].items():
                    print(f"{model} {recipe}: MSE={score['mse']:.6f} | MAE={100 * score['mae']:.2f} points")
        else:
            summary = confirm(args.frozen, args.features, files, args.output)
            for model, score in summary["models"].items():
                gain = score["relative_mse_gain_percent"]
                shown = "undefined" if gain is None else f"{gain:+.2f}%"
                print(f"{model}: MSE gain={shown} | 95% CI={score['ci95_percent']} | {score['decision']}")
        print(f"Saved to {args.output}")
    except (OSError, ValueError, KeyError, TypeError, csv.Error, json.JSONDecodeError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
