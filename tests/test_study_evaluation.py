from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from build_features import PHONEMES
from study_evaluation import (
    MODELS, SPEC, bootstrap_summary, check_confirmation_cohort, choose_alpha,
    feature_matrix, fit_model,
    freeze_digest, learn_features, load_frozen, predict_model, runtime_versions,
    select_block,
)


def clips_for_tests(count=12):
    clips = []
    for index in range(count):
        clips.append({
            "audio_id": f"clip-{index}", "speaker_id": f"speaker-{index}",
            "chapter_id": f"chapter-{index}", "reference_text": "A WORD",
            "words": ["a", "word"], "num_words": 2, "normalized_num_words": 2,
            "mean_word_length": 2.5, "cmudict_missing_proportion": 0,
            "phoneme_proportions": [index / count, 1 - index / count] + [0] * (len(PHONEMES) - 2),
            "num_phonemes": 10, "num_pairs": 10,
            "bigram_counts": {"AA|B": 10},
            "long_word_share": index / count, "noninitial_stress_share": 0,
        })
    return clips


class StudyEvaluationTests(unittest.TestCase):
    def test_feature_learning_and_model_fits_use_training_speakers_only(self):
        training = clips_for_tests()
        held_out = clips_for_tests(1)
        held_out[0]["words"] = ["never_seen_here"]
        held_out[0]["bigram_counts"] = {"T|R": 100000}
        held_out[0]["num_pairs"] = 100000
        state = learn_features(training, "bigrams")
        self.assertEqual(state["bigrams"], ["AA|B"])
        self.assertNotIn("never_seen_here", state["word_counts"])
        before = deepcopy(state)
        values = feature_matrix(held_out, state)
        denominator = state["word_total"] + len(state["word_counts"]) + 1
        self.assertAlmostEqual(values[0, 3], math.log(denominator))
        self.assertEqual(values[0, -1], 0)
        self.assertEqual(state, before)
        self.assertIsNone(learn_features(training[:4], "bigrams"))

        targets = [clip["long_word_share"] for clip in training]
        fitted = fit_model(training, targets, "stress", alpha=1)
        original = deepcopy(fitted)
        first = predict_model(held_out, fitted)
        held_out[0]["words"] = ["another_unseen_word"]
        held_out[0]["phoneme_proportions"] = [0, 1] + [0] * 37
        self.assertEqual(predict_model(held_out, fitted), first)
        self.assertEqual(fitted, original)
        np.testing.assert_allclose(
            fitted["scaler_mean"], feature_matrix(training, fitted).mean(axis=0),
        )
        with patch("study_evaluation.fit_model", wraps=fit_model) as calls:
            alpha = choose_alpha(training, targets, "stress")
        self.assertIn(alpha, SPEC["alphas"])
        self.assertEqual(len(calls.call_args_list), 9)
        for call in calls.call_args_list:
            self.assertLess(len(call.args[0]), len(training))
            self.assertEqual(len({clip["speaker_id"] for clip in call.args[0]}), 8)
        predictions = predict_model(training, fitted)
        self.assertTrue(all(math.isfinite(value) and value >= 0 for value in predictions))
        self.assertLess(
            sum((prediction - target) ** 2 for prediction, target in zip(predictions, targets)),
            sum((sum(targets) / len(targets) - target) ** 2 for target in targets),
        )

    def test_bootstrap_resamples_whole_speakers_with_identical_draws(self):
        rows = [
            {"speaker_id": speaker, "actual_wer": 0.0,
             "lexical_prediction": error, "phonetic_prediction": error / 2}
            for speaker, error in (("one", 1.0), ("two", 2.0), ("three", 3.0))
        ]
        rows.append(dict(rows[0]))
        predictions = {model: deepcopy(rows) for model in MODELS}
        summary = bootstrap_summary(predictions, draws=500, seed=506)
        self.assertEqual(summary, bootstrap_summary(predictions, draws=500, seed=506))
        first = summary["models"]["base.en"]
        self.assertAlmostEqual(first["speaker_balanced_lexical_mse"], 14 / 3)
        self.assertAlmostEqual(first["relative_mse_gain_percent"], 75)
        np.testing.assert_allclose(first["ci95_percent"], [75, 75])
        for model in MODELS:
            self.assertEqual(summary["models"][model], first)
        self.assertTrue(summary["across_models_positive"])

        for model in MODELS:
            for row in predictions[model]:
                row["lexical_prediction"] = row["phonetic_prediction"] = 0
        zero = bootstrap_summary(predictions, draws=100, seed=506)
        self.assertIsNone(zero["models"]["base.en"]["relative_mse_gain_percent"])
        self.assertIsNone(zero["models"]["base.en"]["ci95_percent"])
        self.assertEqual(zero["models"]["base.en"]["decision"], "inconclusive")
        self.assertFalse(zero["across_models_positive"])

    def test_freeze_rejects_a_different_executing_source_before_loading_data(self):
        manifest = {
            "format_version": 1, "spec": SPEC, "versions": runtime_versions(),
            "code": {"study_evaluation.py": {"path": "old_copy.py", "sha256": "old"}},
        }
        manifest["freeze_sha256"] = freeze_digest(manifest)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "frozen.json"
            source.write_text(json.dumps(manifest), encoding="utf-8")
            with patch("study_evaluation.code_records", return_value={"changed": {}}):
                with patch("study_evaluation.check_inputs") as read_inputs:
                    with self.assertRaisesRegex(ValueError, "currently executing"):
                        load_frozen(source)
                    read_inputs.assert_not_called()

        cohort = clips_for_tests(2)
        manifest = {
            "training_speakers": ["development_reader"],
            "historical_speakers": ["development_reader", "earlier_pilot_reader"],
            "historical_ids": ["earlier_pilot_clip"],
        }
        with patch.dict(SPEC, confirmation_clips=2, confirmation_speakers=2, confirmation_clips_per_speaker=1):
            cohort[0]["speaker_id"] = "earlier_pilot_reader"
            with self.assertRaisesRegex(ValueError, "historical speakers"):
                check_confirmation_cohort(cohort, manifest)
            cohort[0]["speaker_id"] = "new_reader"
            cohort[0]["audio_id"] = "earlier_pilot_clip"
            with self.assertRaisesRegex(ValueError, "historical IDs"):
                check_confirmation_cohort(cohort, manifest)
            cohort[0]["audio_id"] = "new_clip"
            check_confirmation_cohort(cohort, manifest)

    def test_selection_ties_favor_smaller_blocks_and_keep_small_gains_honest(self):
        clips = clips_for_tests()
        summary = {"scores": {block: {"mse": 1} for block in ("proportions", "bigrams", "stress")}}
        self.assertEqual(select_block(clips, summary), "bigrams")
        rows = [
            {"speaker_id": str(index), "actual_wer": 0.0,
             "lexical_prediction": 1.0, "phonetic_prediction": math.sqrt(0.98)}
            for index in range(10)
        ]
        result = bootstrap_summary({model: rows for model in MODELS}, draws=100, seed=506)
        score = result["models"]["base.en"]
        self.assertAlmostEqual(score["relative_mse_gain_percent"], 2)
        self.assertEqual(score["decision"], "meaningful_gain_bounded_below_target")
        self.assertTrue(score["small_positive_gain"])
        self.assertFalse(result["across_models_positive"])


if __name__ == "__main__":
    unittest.main()
