import csv
from pathlib import Path
import tempfile
import unittest

from build_features import export_features, extract_features


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.row = {
            "audio_id": "clip-01",
            "speaker_id": "0042",
            "reference_text": "A CAT CAN'T XYZZY",
            "predicted_text": "A different prediction",
            "wer": "0.25",
            "phonemes": "AH K AE T K AE N T",
            "num_phonemes": "8",
            "g2p_words": "can't",
            "unresolved_words": "xyzzy",
        }

    def write_input(self, path, rows):
        with path.open("w", encoding="utf-8-sig", newline="") as source:
            writer = csv.DictWriter(source, fieldnames=self.row)
            writer.writeheader()
            writer.writerows(rows)

    def test_hand_counted_features_and_no_prediction_leakage(self):
        features = extract_features(self.row)
        self.assertEqual(features["num_words"], 4)
        self.assertEqual(features["num_phonemes"], 8)
        self.assertEqual(features["phonemes_per_word"], 2)
        self.assertEqual(features["num_g2p_words"], 1)
        self.assertEqual(features["num_unresolved_words"], 1)
        self.assertEqual(features["vowel_proportion"], 3 / 8)
        self.assertEqual(features["AE_count"], 2)
        self.assertEqual(features["K_proportion"], 2 / 8)
        self.assertEqual(features["ZH_count"], 0)
        self.assertEqual(sum(value for key, value in features.items() if key.endswith("_count")), 8)
        self.assertAlmostEqual(
            sum(value for key, value in features.items() if key.endswith("_proportion")
                and key != "vowel_proportion"), 1,
        )

        changed = extract_features(dict(self.row, predicted_text="Something else", wer="1.25"))
        self.assertEqual(changed.pop("wer"), "1.25")
        features.pop("wer")
        self.assertEqual(features, changed)

    def test_rejects_unusable_pronunciations_and_wer(self):
        cases = [
            ({"phonemes": "AH XX"}, "unrecognized phonemes"),
            ({"num_phonemes": "7"}, "does not match"),
            ({"reference_text": ""}, "must not be empty"),
            ({"phonemes": "", "num_phonemes": "0"}, "must not be empty"),
            ({"wer": "nan"}, "finite, non-negative"),
            ({"wer": "-0.1"}, "finite, non-negative"),
        ]
        for changes, message in cases:
            with self.subTest(changes=changes):
                with self.assertRaisesRegex(ValueError, message):
                    extract_features(dict(self.row, **changes))

    def test_csv_export_preserves_rows_and_protects_existing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.csv"
            output = Path(directory) / "outputs" / "features.csv"
            rows = [self.row, dict(self.row, audio_id="clip-02", wer="1.25")]
            self.write_input(source, rows)
            source_before = source.read_bytes()

            self.assertEqual(export_features(source, output), 2)
            with output.open(encoding="utf-8", newline="") as result:
                saved = list(csv.DictReader(result))
            for original, actual in zip(rows, saved):
                for column in ("audio_id", "speaker_id", "wer"):
                    self.assertEqual(actual[column], original[column])
            self.assertNotIn("predicted_text", saved[0])
            output_before = output.read_bytes()
            for destination in (output, source):
                with self.assertRaises(FileExistsError):
                    export_features(source, destination)
            self.assertEqual(source.read_bytes(), source_before)
            self.assertEqual(output.read_bytes(), output_before)

    def test_bad_csv_fails_before_writing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            cases = {
                "empty": ("", "Missing columns"),
                "missing_column": ("audio_id\nclip-01\n", "Missing columns"),
            }
            for name, (content, message) in cases.items():
                source = directory / f"{name}.csv"
                source.write_text(content, encoding="utf-8")
                output = directory / f"{name}_features.csv"
                with self.assertRaisesRegex(ValueError, message):
                    export_features(source, output)
                self.assertFalse(output.exists())

            for name, rows, message in [
                ("header_only", [], "no clips"),
                ("duplicate", [self.row, self.row], "clip-01: duplicate audio_id"),
                ("truncated", [dict(self.row, unresolved_words=None)], "clip-01: wrong number"),
            ]:
                source = directory / f"{name}.csv"
                self.write_input(source, rows)
                if name == "truncated":
                    lines = source.read_text(encoding="utf-8-sig").splitlines()
                    lines[-1] = lines[-1].rsplit(",", 1)[0]
                    source.write_text("\n".join(lines) + "\n", encoding="utf-8")
                output = directory / f"{name}_features.csv"
                with self.assertRaisesRegex(ValueError, message):
                    export_features(source, output)
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
