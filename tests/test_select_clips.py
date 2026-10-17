from collections import Counter
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

from select_clips import load_clips, save_selection, select_clips


def sample_clips(counts):
    return [
        {
            "dataset": "dev-other",
            "audio_id": f"{speaker}-1-{index:04}",
            "speaker_id": speaker,
            "chapter_id": "1",
            "reference_text": f"EXAMPLE {index}, WITH A COMMA",
        }
        for speaker, count in counts.items()
        for index in range(count)
    ]


class SelectionTests(unittest.TestCase):
    def test_selection_is_balanced_repeatable_and_filters_speakers(self):
        clips = sample_clips({"10": 12, "20": 12, "30": 12})
        original = [dict(clip) for clip in clips]
        selected = select_clips(clips, ["10", "20"], num_clips=9)
        self.assertEqual(len(selected), 9)
        self.assertEqual(len({clip["audio_id"] for clip in selected}), 9)
        self.assertEqual(sorted(Counter(clip["speaker_id"] for clip in selected).values()), [4, 5])
        self.assertEqual({clip["speaker_id"] for clip in selected}, {"10", "20"})
        self.assertEqual(selected, select_clips(list(reversed(clips)), ["20", "10"], 9))
        self.assertNotEqual(selected, select_clips(clips, ["10", "20"], 9, seed=43))
        self.assertEqual(clips, original)

        limited = sample_clips({"10": 1, "20": 2})
        self.assertEqual(Counter(clip["speaker_id"] for clip in select_clips(limited, ["10", "20"], 3)),
                         {"10": 1, "20": 2})

    def test_missing_speakers_and_insufficient_clips_fail(self):
        clips = sample_clips({"10": 2, "20": 10})
        for speakers, count, message in [
            (["10", "99"], 4, "Speaker IDs not found"),
            (["10", "20"], 20, "Cannot select 20 clips evenly"),
            (["10", "20"], 8, "Cannot select 8 clips evenly"),
            (["10", "20"], 1, "at least one clip per"),
            ([], 10, "at least one speaker"),
        ]:
            with self.subTest(speakers=speakers, count=count):
                with self.assertRaisesRegex(ValueError, message):
                    select_clips(clips, speakers, count)
        with self.assertRaisesRegex(ValueError, "Duplicate audio ID"):
            select_clips(clips + [clips[0]], ["10"], 1)

    def test_metadata_pages_and_saved_selection(self):
        clips = sample_clips({"10": 2, "20": 2})
        pages = []
        for offset in (0, 2):
            rows = [{"row": {
                "id": clip["audio_id"],
                "speaker_id": int(clip["speaker_id"]),
                "chapter_id": 1,
                "text": clip["reference_text"],
                "audio": [{"src": "https://example.invalid/do-not-download.flac"}],
            }} for clip in clips[offset:offset + 2]]
            pages.append(io.BytesIO(json.dumps({
                "rows": rows, "num_rows_total": 4, "partial": False,
            }).encode("utf-8")))
        with patch("select_clips.urlopen", side_effect=pages) as request:
            self.assertEqual(load_clips("dev-other"), clips)
        self.assertEqual(request.call_count, 2)
        for call, offset in zip(request.call_args_list, ("0", "2")):
            url = urlparse(call.args[0])
            self.assertEqual((url.netloc, url.path), ("datasets-server.huggingface.co", "/rows"))
            query = parse_qs(url.query)
            self.assertEqual(query["config"], ["other"])
            self.assertEqual(query["split"], ["validation"])
            self.assertEqual(query["offset"], [offset])

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "outputs" / "selection.csv"
            selected = select_clips(clips, ["10", "20"], 4)
            save_selection(selected, output)
            with output.open(encoding="utf-8", newline="") as source:
                self.assertEqual(list(csv.DictReader(source)), selected)
            original = output.read_bytes()
            with self.assertRaises(FileExistsError):
                save_selection(selected, output)
            self.assertEqual(output.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
