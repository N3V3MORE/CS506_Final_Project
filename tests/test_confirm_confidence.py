from copy import deepcopy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import confirm_confidence as confirmation
from evaluate_confidence import feature_matrix, read_confidence
from transcribe_selection import CONFIDENCE_METHOD, NORMALIZATION, RESULT_COLUMNS, TRANSCRIPTION_SETTINGS


DICTIONARY = {"a": [], "cat": []}


def write_results(path, rows, selection_hash="training"):
    with path.open("x", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=RESULT_COLUMNS + ["avg_logprob"])
        writer.writeheader()
        writer.writerows(rows)
    path.with_suffix(".metadata.json").write_text(json.dumps({
        "model_name": "base.en", "device": "cuda:0", "effective_device": "cuda:0",
        "confidence": CONFIDENCE_METHOD, "normalization": NORMALIZATION,
        "settings": TRANSCRIPTION_SETTINGS, "selection_sha256": selection_hash,
        "versions": {"python": "fixture", "openai-whisper": "fixture", "torch": "fixture", "jiwer": "fixture"},
    }), encoding="utf-8")


def prepare(directory):
    training = [
        dict(audio_id=f"{10+i//2}-1-{i:04}", speaker_id=str(10+i//2),
             reference_text="A CAT", predicted_text="a cat", wer=i/19, avg_logprob=-i/19)
        for i in range(20)
    ]
    new = [
        dict(audio_id=f"{100+i//2}-2-{i:04}", speaker_id=str(100+i//2),
             reference_text="A CAT", predicted_text="a cat", wer=i/3, avg_logprob=-i/3)
        for i in range(4)
    ]
    train_path, history_path, selection_path = [directory/name for name in ("train.csv", "history.csv", "selection.csv")]
    write_results(train_path, training)
    history_path.write_text("dataset,audio_id,speaker_id\ntrain-other-500,90-1-0000,90\n", encoding="utf-8")
    with selection_path.open("x", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=["dataset", "audio_id", "speaker_id", "chapter_id", "reference_text"])
        writer.writeheader()
        writer.writerows({
            "dataset": "train-other-500", "audio_id": row["audio_id"], "speaker_id": row["speaker_id"],
            "chapter_id": "2", "reference_text": row["reference_text"],
        } for row in new)
    return training, new, train_path, history_path, selection_path


class FrozenConfidenceTests(unittest.TestCase):
    def test_saved_fit_matches_sklearn_and_new_labels_are_read_last(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(confirmation.PROTOCOL, clips=4, speakers=2, clips_per_speaker=2):
            directory = Path(temporary)
            training, new, train_path, history_path, selection_path = prepare(directory)
            frozen_path = directory/"frozen.json"
            frozen = confirmation.freeze([train_path], history_path, selection_path, frozen_path, DICTIONARY)
            new_path = directory/"new.csv"
            write_results(new_path, new, confirmation.file_record(selection_path)["sha256"])
            features = read_confidence([new_path], include_targets=False)
            self.assertTrue(all("wer" not in row and "reference_text" not in row for row in features))
            actual = confirmation.predict_frozen(features, frozen, DICTIONARY)
            train_values = feature_matrix(training, range(len(training)), DICTIONARY)
            new_values = confirmation.transform_features(features, frozen["feature_state"], DICTIONARY)
            for column, width in (("text_prediction", 5), ("confidence_prediction", 6)):
                expected = make_pipeline(StandardScaler(), Ridge(alpha=1)).fit(
                    [row[:width] for row in train_values], [row["wer"] for row in training],
                ).predict([row[:width] for row in new_values])
                np.testing.assert_allclose([row[column] for row in actual], np.maximum(expected, 0))
            def check_reader(paths, include_targets=True):
                if include_targets:
                    self.assertTrue(predict.called)
                return read_confidence(paths, include_targets=include_targets)
            with patch("confirm_confidence.predict_frozen", wraps=confirmation.predict_frozen) as predict, \
                 patch("confirm_confidence.read_confidence", side_effect=check_reader), \
                 patch("confirm_confidence.Ridge", side_effect=AssertionError("No fresh-data fitting")):
                predictions, scores, interval = confirmation.evaluate(
                    frozen_path, [new_path], directory/"predictions.csv", DICTIONARY,
                )
            self.assertEqual(len(predictions), 4)
            self.assertEqual(len(interval), 2)
            changed = deepcopy(new)
            for row in changed:
                row["wer"] += 10
            changed_path = directory/"changed.csv"
            write_results(changed_path, changed, confirmation.file_record(selection_path)["sha256"])
            repeated, _, _ = confirmation.evaluate(frozen_path, [changed_path], directory/"repeated.csv", DICTIONARY)
            for before, after in zip(predictions, repeated):
                for key in ("mean_prediction", "median_prediction", "text_prediction", "confidence_prediction"):
                    self.assertEqual(before[key], after[key])

    def test_frozen_inputs_dictionary_sources_and_historical_speakers_are_protected(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(confirmation.PROTOCOL, clips=4, speakers=2, clips_per_speaker=2):
            directory = Path(temporary)
            training, new, train_path, history_path, selection_path = prepare(directory)
            frozen_path = directory/"frozen.json"
            frozen = confirmation.freeze([train_path], history_path, selection_path, frozen_path, DICTIONARY)
            with self.assertRaises(FileExistsError):
                confirmation.freeze([train_path], history_path, selection_path, frozen_path, DICTIONARY)
            with self.assertRaisesRegex(ValueError, "dictionary"):
                confirmation.load_frozen(frozen_path, {"different": []})
            with patch("confirm_confidence.source_hashes", return_value={}):
                with self.assertRaisesRegex(ValueError, "Source"):
                    confirmation.load_frozen(frozen_path, DICTIONARY)
            for row in new[:2]:
                row["speaker_id"] = "90"
                row["audio_id"] = row["audio_id"].replace("100-", "90-")
            with self.assertRaisesRegex(ValueError, "historical"):
                confirmation.check_cohort(new, frozen["excluded_speakers"], frozen["excluded_ids"], frozen["protocol"])
            train_path.write_text(train_path.read_text()+"\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Frozen input changed"):
                confirmation.load_frozen(frozen_path, DICTIONARY)


if __name__ == "__main__":
    unittest.main()
