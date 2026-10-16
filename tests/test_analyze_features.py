import csv
from pathlib import Path
import tempfile
import unittest

from analyze_features import analyze_features, read_features, summarize_phonemes
from build_features import PHONEMES


def sample_clips(levels):
    clips = []
    for index, level in enumerate(levels):
        row = {
            "audio_id": f"clip-{index}",
            "speaker_id": "0042",
            "wer": level / 10,
            "num_phonemes": 10,
            "num_g2p_words": 0,
            "num_unresolved_words": 0,
        }
        for phone in PHONEMES:
            count = level if phone == "AA" else 10 - level if phone == "T" else 0
            row[f"{phone}_count"] = count
            row[f"{phone}_proportion"] = count / 10
        clips.append(row)
    return clips


def write_input(path, clips):
    with path.open("w", encoding="utf-8-sig", newline="") as source:
        writer = csv.DictWriter(source, fieldnames=clips[0])
        writer.writeheader()
        writer.writerows(clips)


class AssociationTests(unittest.TestCase):
    def test_known_correlations_and_group_averages(self):
        clips = sample_clips([0, 0, 0, 0, 0, 1, 2, 3, 4, 5])
        results = {row["phoneme"]: row for row in summarize_phonemes(clips)}
        self.assertEqual(len(results), 39)
        aa = results["AA"]
        self.assertAlmostEqual(aa["proportion_wer_correlation"], 1)
        self.assertEqual(aa["clips_with_phoneme"], 5)
        self.assertEqual(aa["clips_without_phoneme"], 5)
        self.assertEqual(aa["total_occurrences"], 15)
        self.assertAlmostEqual(aa["mean_clip_wer_with"], 0.3)
        self.assertEqual(aa["mean_clip_wer_without"], 0)
        self.assertAlmostEqual(aa["wer_difference"], 0.3)
        self.assertEqual(aa["status"], "exploratory")
        self.assertAlmostEqual(results["T"]["proportion_wer_correlation"], -1)
        self.assertIsNone(results["T"]["mean_clip_wer_without"])
        self.assertIsNone(results["T"]["wer_difference"])

    def test_rare_and_constant_sounds_are_not_ranked_as_supported(self):
        clips = sample_clips([0] * 9 + [1])
        results = summarize_phonemes(clips)
        aa = next(row for row in results if row["phoneme"] == "AA")
        self.assertEqual(aa["status"], "low_support")
        self.assertAlmostEqual(aa["proportion_wer_correlation"], 1)
        self.assertEqual(results[0]["phoneme"], "T")
        absent = next(row for row in results if row["phoneme"] == "CH")
        self.assertIsNone(absent["proportion_wer_correlation"])
        self.assertIsNone(absent["mean_clip_wer_with"])
        for clip in clips:
            clip["wer"] = 0.2
        constant = summarize_phonemes(clips)
        self.assertTrue(all(row["proportion_wer_correlation"] is None for row in constant))
        self.assertTrue(all(row["status"] == "constant_wer" for row in constant))

    def test_csv_validation_and_file_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            clips = sample_clips([0, 0, 0, 0, 0, 1, 2, 3, 4, 5])
            source = directory / "features.csv"
            output = directory / "outputs" / "associations.csv"
            write_input(source, clips)
            source_before = source.read_bytes()
            saved_clips, results = analyze_features(source, output)
            self.assertEqual(len(saved_clips), 10)
            with output.open(encoding="utf-8", newline="") as stream:
                saved = list(csv.DictReader(stream))
            self.assertEqual(len(saved), 39)
            self.assertEqual({row["speaker_id"] for row in saved_clips}, {"0042"})
            absent = next(row for row in saved if row["phoneme"] == "CH")
            self.assertEqual(absent["proportion_wer_correlation"], "")
            self.assertEqual(absent["mean_clip_wer_with"], "")
            output_before = output.read_bytes()
            for destination in (source, output):
                with self.assertRaises(FileExistsError):
                    analyze_features(source, destination)
            self.assertEqual(source.read_bytes(), source_before)
            self.assertEqual(output.read_bytes(), output_before)

            for name, changes, message in [
                ("bad_wer", {"wer": "nan"}, "clip-0: wer must be finite"),
                ("bad_proportion", {"AA_proportion": 0.4}, "clip-0: AA_proportion"),
                ("bad_total", {"num_phonemes": 9}, "clip-0: phoneme counts"),
                ("duplicate", {"audio_id": "clip-1"}, "clip-1: duplicate audio_id"),
            ]:
                invalid = [dict(clip) for clip in clips]
                invalid[0].update(changes)
                bad_source = directory / f"{name}.csv"
                bad_output = directory / f"{name}_out.csv"
                write_input(bad_source, invalid)
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, message):
                    analyze_features(bad_source, bad_output)
                self.assertFalse(bad_output.exists())
            empty = directory / "empty.csv"
            empty.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Missing columns"):
                read_features(empty)
            one = directory / "one.csv"
            write_input(one, clips[:1])
            with self.assertRaisesRegex(ValueError, "At least two clips"):
                read_features(one)


if __name__ == "__main__":
    unittest.main()
