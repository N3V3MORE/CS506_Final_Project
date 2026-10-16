import csv
from pathlib import Path
import tempfile
import unittest

from build_features import PHONEMES
from evaluate_features import evaluate_clips, evaluate_features, prediction_errors


def sample_clips(count=25):
    clips = []
    for index in range(count):
        row = {
            "audio_id": f"clip-{index}",
            "speaker_id": "0042",
            "wer": 1.5 * index / (count - 1),
            "num_phonemes": count - 1,
            "num_g2p_words": 0,
            "num_unresolved_words": 0,
        }
        for phone in PHONEMES:
            occurrences = 0
            if phone == "AA":
                occurrences = index
            elif phone == "T":
                occurrences = count - 1 - index
            row[f"{phone}_count"] = occurrences
            row[f"{phone}_proportion"] = occurrences / (count - 1)
        clips.append(row)
    return clips


class PredictionTests(unittest.TestCase):
    def test_held_out_targets_do_not_change_their_predictions(self):
        clips = sample_clips()
        original = evaluate_clips(clips)
        held_out_ids = {row["audio_id"] for row in original if row["fold"] == 1}
        changed = [dict(clip) for clip in clips]
        for clip in changed:
            if clip["audio_id"] in held_out_ids:
                clip["wer"] += 100
        repeated = evaluate_clips(changed)
        training = [clip["wer"] for clip in clips if clip["audio_id"] not in held_out_ids]
        expected_mean = sum(training) / len(training)
        for before, after in zip(original, repeated):
            if before["audio_id"] in held_out_ids:
                self.assertEqual(before["fold"], after["fold"])
                self.assertAlmostEqual(before["baseline_prediction"], expected_mean)
                self.assertEqual(before["baseline_prediction"], after["baseline_prediction"])
                self.assertAlmostEqual(before["phoneme_prediction"], after["phoneme_prediction"])

    def test_known_signal_beats_mean_and_saved_files_are_protected(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "features.csv"
            output = directory / "outputs" / "predictions.csv"
            clips = sample_clips()
            with source.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=clips[0])
                writer.writeheader()
                writer.writerows(clips)
            original = source.read_bytes()
            predictions, scores = evaluate_features(source, output)
            self.assertEqual([row["audio_id"] for row in predictions],
                             [clip["audio_id"] for clip in clips])
            self.assertEqual(len({row["fold"] for row in predictions}), 5)
            for metric in ("mae", "mse"):
                self.assertLess(scores["phoneme_prediction"][metric],
                                scores["baseline_prediction"][metric] / 5)
            self.assertGreater(max(row["phoneme_prediction"] for row in predictions), 1)
            self.assertTrue(all(row["phoneme_prediction"] >= 0 for row in predictions))
            with output.open(encoding="utf-8", newline="") as stream:
                saved = list(csv.DictReader(stream))
            self.assertEqual(len(saved), len(clips))
            output_before = output.read_bytes()
            for destination in (source, output):
                with self.assertRaises(FileExistsError):
                    evaluate_features(source, destination)
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(output.read_bytes(), output_before)

    def test_speakers_stay_together_and_constant_targets_are_valid(self):
        clips = sample_clips(12)
        for index, clip in enumerate(clips):
            clip["speaker_id"] = f"speaker-{index // 4}"
            clip["wer"] = 0.25
        predictions = evaluate_clips(clips)
        speaker_folds = {}
        for row in predictions:
            speaker_folds.setdefault(row["speaker_id"], set()).add(row["fold"])
            self.assertEqual(row["split_strategy"], "speaker_grouped")
            self.assertAlmostEqual(row["baseline_prediction"], 0.25)
            self.assertAlmostEqual(row["phoneme_prediction"], 0.25)
        self.assertEqual(len({row["fold"] for row in predictions}), 3)
        self.assertTrue(all(len(folds) == 1 for folds in speaker_folds.values()))
        scores = prediction_errors(predictions)
        self.assertEqual(scores["baseline_prediction"]["mse"], 0)
        self.assertEqual(scores["phoneme_prediction"]["mse"], 0)
        with self.assertRaisesRegex(ValueError, "At least five clips"):
            evaluate_clips(clips[:4])


if __name__ == "__main__":
    unittest.main()
