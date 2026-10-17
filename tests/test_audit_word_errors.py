import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from audit_word_errors import audit_results
from transcribe_selection import RESULT_COLUMNS


def write_results(path, rows):
    with path.open("x", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


class WordAuditTests(unittest.TestCase):
    def test_word_errors_and_insertions_do_not_cross_changed_tokenization(self):
        rows = [
            {"audio_id": "1-1-1", "speaker_id": "1", "reference_text": "A ZORG CAT",
             "predicted_text": "A DOG CAT EXTRA", "wer": 2 / 3},
            {"audio_id": "2-1-1", "speaker_id": "2", "reference_text": "ONE CAT",
             "predicted_text": "CAT", "wer": .5},
            {"audio_id": "3-1-1", "speaker_id": "3", "reference_text": "21 CATS",
             "predicted_text": "TWENTY ONE CATS", "wer": 0},
        ]
        normalize = lambda text: text.lower().replace("twenty one", "21")
        dictionary = {word: [] for word in ("a", "cat", "one", "cats")}
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "results.csv"
            write_results(source, rows)
            before = source.read_bytes()
            with patch("audit_word_errors.normalize_text", side_effect=normalize):
                clips, words, summary = audit_results(source, dictionary)
            self.assertEqual(summary["eligible_clips"], 2)
            self.assertEqual(summary["insertions"], 1)
            self.assertEqual(summary["missing_word_count"], 1)
            self.assertEqual(summary["missing_word_errors"], 1)
            self.assertEqual(summary["covered_word_count"], 4)
            self.assertEqual(summary["covered_word_errors"], 1)
            self.assertAlmostEqual(summary["pooled_wer"], 3 / 7)
            self.assertFalse(clips[2]["word_audit_eligible"])
            self.assertNotIn("3-1-1", {word["audio_id"] for word in words})
            self.assertEqual(source.read_bytes(), before)

    def test_multiword_substitutions_show_corresponding_words(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "results.csv"
            write_results(source, [{"audio_id": "1-1-1", "speaker_id": "1",
                                  "reference_text": "CAT DOG", "predicted_text": "HAT FOG", "wer": 1}])
            with patch("audit_word_errors.normalize_text", side_effect=str.lower):
                _, words, _ = audit_results(source, {"cat": [], "dog": []})
            self.assertEqual([(row["reference_word"], row["predicted_words"]) for row in words],
                             [("cat", "hat"), ("dog", "fog")])
            self.assertEqual({row["alignment_scope"] for row in words}, {"word"})

    def test_incompatible_scores_and_duplicate_ids_are_rejected(self):
        row = {"audio_id": "1-1-1", "speaker_id": "1", "reference_text": "CAT",
               "predicted_text": "DOG", "wer": 0}
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "wrong-score.csv"
            write_results(source, [row])
            with patch("audit_word_errors.normalize_text", side_effect=str.lower):
                with self.assertRaisesRegex(ValueError, "saved WER does not match"):
                    audit_results(source, {"cat": []})
            source = Path(directory) / "duplicates.csv"
            row["wer"] = 1
            write_results(source, [row, row])
            with patch("audit_word_errors.normalize_text", side_effect=str.lower):
                with self.assertRaisesRegex(ValueError, "duplicate"):
                    audit_results(source, {"cat": []})

    def test_wer_above_one_and_empty_predictions_are_retained(self):
        rows = [
            {"audio_id": "1-1-1", "speaker_id": "1", "reference_text": "CAT",
             "predicted_text": "DOG RAT BAT", "wer": 3},
            {"audio_id": "1-1-2", "speaker_id": "1", "reference_text": "CAT",
             "predicted_text": "", "wer": 1},
        ]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "results.csv"
            write_results(source, rows)
            with patch("audit_word_errors.normalize_text", side_effect=str.lower):
                _, _, summary = audit_results(source, {"cat": []})
            self.assertEqual(summary["wer_over_one_clips"], 1)
            self.assertEqual(summary["covered_word_errors"], 2)
            self.assertEqual(summary["insertions"], 2)
            self.assertEqual(summary["deletions"], 1)


if __name__ == "__main__":
    unittest.main()
