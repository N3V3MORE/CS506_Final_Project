from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError

from transcribe_selection import load_audio_metadata, load_gpu_model, run_selection


def write_selection(path, count=3):
    rows = [
        {
            "dataset": "dev-other", "audio_id": f"10-20-{index:04}",
            "speaker_id": "10", "chapter_id": "20", "reference_text": "A CAT",
        }
        for index in range(count)
    ]
    with path.open("x", encoding="utf-8", newline="") as source:
        writer = csv.DictWriter(source, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def read_results(path):
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


class TranscriptionTests(unittest.TestCase):
    def test_audio_links_are_paged_once_and_incomplete_pages_are_rejected(self):
        load_audio_metadata.cache_clear()
        self.addCleanup(load_audio_metadata.cache_clear)
        pages = [
            {"partial": False, "num_rows_total": 2,
             "rows": [{"row": {"id": f"10-20-{index:04}", "audio": [{"src": f"clip-{index}.flac"}]}}]}
            for index in range(2)
        ]
        responses = [io.BytesIO(json.dumps(page).encode()) for page in pages]
        with patch("transcribe_selection.urlopen", side_effect=responses) as request:
            recordings = load_audio_metadata("dev-other")
            self.assertEqual(set(recordings), {"10-20-0000", "10-20-0001"})
            self.assertEqual(recordings["10-20-0001"]["audio"][0]["src"], "clip-1.flac")
            self.assertIs(load_audio_metadata("dev-other"), recordings)
            self.assertEqual(request.call_count, 2)
            self.assertIn("/rows?", request.call_args_list[0].args[0])
            self.assertIn("offset=1", request.call_args_list[1].args[0])
        load_audio_metadata.cache_clear()
        pages[0]["partial"] = True
        with patch("transcribe_selection.urlopen", return_value=io.BytesIO(json.dumps(pages[0]).encode())):
            with self.assertRaisesRegex(ValueError, "incomplete audio metadata"):
                load_audio_metadata("dev-other")

    def test_one_clip_limit_then_resume_without_retranscribing(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "selection.csv"
            output = Path(directory) / "outputs" / "results.csv"
            clips = write_selection(source)
            source_before = source.read_bytes()
            model = Mock()
            model.transcribe.return_value = {"text": "A cat."}
            with patch("transcribe_selection.load_gpu_model", return_value=model) as load_model, \
                 patch("transcribe_selection.fetch_audio", return_value=[0.0]) as fetch, \
                 patch("transcribe_selection.score_transcription", return_value=0.0), \
                 redirect_stdout(io.StringIO()):
                self.assertEqual(run_selection(source, output, limit=1), (1, 0))
                first_row = read_results(output)[0]
                self.assertEqual(first_row["audio_id"], clips[0]["audio_id"])
                self.assertEqual(first_row["predicted_text"], "A cat.")
                self.assertEqual(first_row["wer"], "0.0")
                self.assertEqual(model.transcribe.call_args.kwargs, {"language": "en", "fp16": False})
                self.assertEqual(run_selection(source, output, limit=1), (0, 0))
                self.assertEqual(load_model.call_count, 1)
                self.assertEqual(fetch.call_count, 1)
                self.assertEqual(run_selection(source, output), (2, 0))
                self.assertEqual(load_model.call_count, 2)
                self.assertEqual(fetch.call_count, 3)
            saved = read_results(output)
            self.assertEqual(saved[0], first_row)
            self.assertEqual([row["audio_id"] for row in saved], [clip["audio_id"] for clip in clips])
            self.assertEqual(source.read_bytes(), source_before)

    def test_failed_audio_is_retried_and_disk_errors_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "selection.csv"
            output = Path(directory) / "results.csv"
            clips = write_selection(source, 2)
            model = Mock()
            model.transcribe.return_value = {"text": "A cat."}
            with patch("transcribe_selection.load_gpu_model", return_value=model), \
                 patch("transcribe_selection.fetch_audio") as fetch, \
                 patch("transcribe_selection.score_transcription", return_value=0.0), \
                 redirect_stdout(io.StringIO()):
                fetch.side_effect = [ValueError("bad audio"), [0.0]]
                self.assertEqual(run_selection(source, output), (1, 1))
                self.assertEqual([row["audio_id"] for row in read_results(output)], [clips[1]["audio_id"]])
                fetch.side_effect = None
                fetch.return_value = [0.0]
                self.assertEqual(run_selection(source, output), (1, 0))
                self.assertEqual(model.transcribe.call_count, 2)
                self.assertEqual({row["audio_id"] for row in read_results(output)},
                                 {clip["audio_id"] for clip in clips})
                with patch("transcribe_selection.csv.DictWriter.writerow", side_effect=OSError("disk full")):
                    with self.assertRaisesRegex(OSError, "disk full"):
                        run_selection(source, Path(directory) / "disk-error.csv")
                fetch.side_effect = HTTPError("https://example.com", 429, "rate limited", {}, None)
                with self.assertRaisesRegex(RuntimeError, "rate limit reached"):
                    run_selection(source, Path(directory) / "rate-limit.csv")

            with patch("transcribe_selection.load_gpu_model", return_value=model), \
                 patch("transcribe_selection.load_audio_metadata", side_effect=OSError("server unavailable")) as metadata, \
                 redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "Could not load audio metadata"):
                    run_selection(source, Path(directory) / "metadata-error.csv")
                metadata.assert_called_once()

    def test_bad_checkpoint_and_missing_cuda_stop_early(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "selection.csv"
            output = Path(directory) / "results.csv"
            write_selection(source)
            output.write_text("wrong,header\n", encoding="utf-8")
            before = output.read_bytes()
            with patch("transcribe_selection.load_gpu_model") as load_model:
                with self.assertRaisesRegex(ValueError, "incompatible CSV header"):
                    run_selection(source, output, limit=1)
                load_model.assert_not_called()
            self.assertEqual(output.read_bytes(), before)

        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        whisper = SimpleNamespace(load_model=Mock())
        with patch.dict(sys.modules, {"torch": torch, "whisper": whisper}):
            with self.assertRaisesRegex(RuntimeError, "CUDA is unavailable"):
                load_gpu_model()
            whisper.load_model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
