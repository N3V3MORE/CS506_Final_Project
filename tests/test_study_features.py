import csv
import json
from pathlib import Path
import tempfile
import unittest

from build_features import PHONEMES
from study_features import export_study_features, extract_study_features


class StudyFeatureTests(unittest.TestCase):
    def setUp(self):
        self.cmu = {
            "banana": [["B", "AH0", "N", "AE1", "N", "AH0"]],
            "table": [["T", "EY1", "B", "AH0", "L"]],
            "cat": [["K", "AE1", "T"]],
        }
        self.g2p = lambda word: ["Z", "IY1", " ", "."] if word == "zzy" else []
        self.normalize = lambda text: text.lower()

    def test_sound_order_stress_and_dictionary_coverage(self):
        features = extract_study_features("BANANA TABLE CAT ZZY ???", self.cmu, self.g2p, self.normalize)
        self.assertEqual(features["words"], ["banana", "table", "cat", "zzy"])
        self.assertEqual(features["num_words"], 4)
        self.assertEqual(features["cmudict_missing_proportion"], 1 / 4)
        self.assertEqual(features["long_word_share"], 1 / 3)
        self.assertEqual(features["noninitial_stress_share"], 1 / 2)
        self.assertEqual(features["num_phonemes"], 16)
        self.assertEqual(features["num_pairs"], 11)
        self.assertEqual(sum(features["bigram_counts"].values()), 11)
        self.assertEqual(features["bigram_counts"]["B|AH"], 2)
        self.assertNotIn("AH|T", features["bigram_counts"])
        self.assertNotIn("Z|IY", features["bigram_counts"])
        self.assertEqual(features["phoneme_proportions"][PHONEMES.index("AH")], 3 / 16)
        self.assertAlmostEqual(sum(features["phoneme_proportions"]), 1)
        self.assertEqual(features["g2p_words"], ["zzy"])

    def test_unresolved_words_and_exact_normalizer_input(self):
        seen = []
        def normalize(text):
            seen.append(text)
            return "one normalized word"
        features = extract_study_features("WELL-KNOWN XYZZY", {}, lambda word: [], normalize)
        self.assertEqual(seen, ["WELLKNOWN XYZZY"])
        self.assertEqual(features["words"], ["well", "known", "xyzzy"])
        self.assertEqual(features["normalized_num_words"], 3)
        self.assertEqual(features["num_phonemes"], 0)
        self.assertEqual(features["phoneme_proportions"], [0.0] * 39)
        self.assertEqual(features["long_word_share"], 0)
        self.assertEqual(features["noninitial_stress_share"], 0)
        self.assertEqual(features["cmudict_missing_proportion"], 1)
        self.assertEqual(features["unresolved_words"], ["well", "known", "xyzzy"])
        for text in ("", "!!!", "123"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                extract_study_features(text, self.cmu, self.g2p, self.normalize)
        with self.assertRaisesRegex(ValueError, "normalized words"):
            extract_study_features("CAT", self.cmu, self.g2p, lambda text: "")

    def test_export_protects_sources_and_rejects_bad_rows_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "selection.csv"
            output = Path(directory) / "features.jsonl"
            row = {"dataset": "dev-other", "audio_id": "42-10-0001", "speaker_id": "42",
                   "chapter_id": "10", "reference_text": "CAT"}
            with source.open("w", encoding="utf-8", newline="") as destination:
                writer = csv.DictWriter(destination, fieldnames=row)
                writer.writeheader()
                writer.writerow(row)
            source_before = source.read_bytes()
            self.assertEqual(export_study_features(source, output, self.cmu, self.g2p, self.normalize), 1)
            saved = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(saved["audio_id"], row["audio_id"])
            self.assertEqual(saved["reference_text"], "CAT")
            self.assertNotIn("wer", saved)
            self.assertNotIn("predicted_text", saved)
            output_before = output.read_bytes()
            for path in (source, output):
                with self.assertRaises(FileExistsError):
                    export_study_features(source, path, self.cmu, self.g2p, self.normalize)
            self.assertEqual(source.read_bytes(), source_before)
            self.assertEqual(output.read_bytes(), output_before)
            invalid_output = Path(directory) / "invalid.jsonl"
            with self.assertRaisesRegex(ValueError, "42-10-0001.*normalized words"):
                export_study_features(source, invalid_output, self.cmu, self.g2p, lambda text: "")
            self.assertFalse(invalid_output.exists())


if __name__ == "__main__":
    unittest.main()
