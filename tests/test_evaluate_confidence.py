from copy import deepcopy
import csv
import json
import math
from pathlib import Path
from statistics import median
import tempfile
import unittest

from evaluate_confidence import (
    CONFIDENCE, PREDICTION_LABELS, evaluate_clips, evaluate_confidence,
    feature_matrix, prediction_errors, read_confidence,
)
from transcribe_selection import NORMALIZATION, RESULT_COLUMNS, TRANSCRIPTION_SETTINGS


def sample_rows():
    return [
        {
            "audio_id": f"clip-{index}", "speaker_id": f"speaker-{index // 2}",
            "reference_text": "REFERENCE WORDS ARE NOT FEATURES",
            "predicted_text": "a cat", "wer": 1.5 * index / 19,
            "avg_logprob": -index / 19,
        }
        for index in range(20)
    ]


def write_input(path, rows):
    with path.open("x", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=RESULT_COLUMNS + ["avg_logprob"])
        writer.writeheader()
        writer.writerows(rows)
    path.with_suffix(".metadata.json").write_text(json.dumps({
        "model_name": "base.en", "device": "cuda:0", "effective_device": "cuda:0",
        "confidence": CONFIDENCE, "normalization": NORMALIZATION, "settings": TRANSCRIPTION_SETTINGS,
    }), encoding="utf-8")


class ConfidenceTests(unittest.TestCase):
    def test_confidence_signal_beats_text_and_reference_words_are_unused(self):
        rows = sample_rows()
        dictionary = {"a": [], "cat": []}
        predictions = evaluate_clips(rows, dictionary)
        scores = prediction_errors(predictions)
        self.assertLess(scores["confidence_prediction"]["mae"], scores["text_prediction"]["mae"] / 5)
        self.assertGreater(max(row["confidence_prediction"] for row in predictions), 1)
        for row in predictions:
            self.assertTrue(all(math.isfinite(row[column]) and row[column] >= 0
                                for column, _ in PREDICTION_LABELS))
        changed = deepcopy(rows)
        for row in changed:
            row["reference_text"] = "completely different reference with many more words"
        self.assertEqual(predictions, evaluate_clips(changed, dictionary))

    def test_speakers_and_training_statistics_stay_separate(self):
        rows = sample_rows()
        dictionary = {"a": [], "cat": []}
        predictions = evaluate_clips(rows, dictionary)
        speaker_folds = {}
        for row in predictions:
            speaker_folds.setdefault(row["speaker_id"], set()).add(row["fold"])
        self.assertTrue(all(len(folds) == 1 for folds in speaker_folds.values()))
        held_out = {row["audio_id"] for row in predictions if row["fold"] == 1}
        changed = deepcopy(rows)
        for row in changed:
            if row["audio_id"] in held_out:
                row["wer"] += 100
        repeated = evaluate_clips(changed, dictionary)
        for before, after in zip(predictions, repeated):
            if before["audio_id"] in held_out:
                for column, _ in PREDICTION_LABELS:
                    self.assertEqual(before[column], after[column])
        train = [index for index, row in enumerate(rows) if row["audio_id"] not in held_out]
        test = next(index for index, row in enumerate(rows) if row["audio_id"] in held_out)
        before = feature_matrix(rows, train, dictionary)
        changed[test]["predicted_text"] = ""
        changed[test]["avg_logprob"] = None
        after = feature_matrix(changed, train, dictionary)
        self.assertEqual([before[index] for index in train], [after[index] for index in train])
        self.assertEqual(after[test][:5], [0, 0, 0, 0, 1])
        self.assertEqual(after[test][-1], median(rows[index]["avg_logprob"] for index in train))

    def test_input_validation_and_existing_outputs_are_protected(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source, output = directory / "confidence.csv", directory / "predictions.csv"
            rows = sample_rows()
            rows[0]["predicted_text"], rows[0]["avg_logprob"] = "", None
            write_input(source, rows)
            original = source.read_bytes()
            loaded, predictions, scores = evaluate_confidence([source], output, {"a": [], "cat": []})
            self.assertEqual(len(predictions), len(rows))
            self.assertIsNone(loaded[0]["avg_logprob"])
            self.assertEqual(source.read_bytes(), original)
            saved = output.read_bytes()
            for target in (source, output):
                with self.assertRaises(FileExistsError):
                    evaluate_confidence([source], target, {})
            self.assertEqual(output.read_bytes(), saved)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                read_confidence([source, source])
            for index, (key, value) in enumerate((("wer", "nan"), ("wer", -1), ("avg_logprob", "nan"), ("avg_logprob", 0.1))):
                invalid = deepcopy(rows)
                invalid[0][key] = value
                path = directory / f"bad-{index}.csv"
                write_input(path, invalid)
                with self.assertRaises(ValueError):
                    read_confidence([path])
            bad_header = directory / "bad-header.csv"
            write_input(bad_header, [])
            bad_header.write_text("wrong,header\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "columns"):
                read_confidence([bad_header])


if __name__ == "__main__":
    unittest.main()
