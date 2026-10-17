import io
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from study_data import RangeFile, download_budget, estimate_audio_bytes, selected_groups


class RangeTests(unittest.TestCase):
    def test_exact_ranges_and_budget_are_enforced(self):
        response = io.BytesIO(b"abcd")
        response.status = 206
        response.headers = {"Content-Range": "bytes 6-9/10"}
        budget = download_budget(4)
        with RangeFile("https://example.invalid", 10, budget) as remote:
            remote.seek(-4, 2)
            with patch("study_data.urlopen", return_value=response) as request:
                self.assertEqual(remote.read(4), b"abcd")
            self.assertEqual(request.call_args.args[0].headers["Range"], "bytes=6-9")
            self.assertEqual(budget["used"], 4)
            remote.seek(0)
            with self.assertRaisesRegex(ValueError, "budget"):
                remote.read(1)
            with self.assertRaisesRegex(ValueError, "Whole-file"):
                remote.read()

    def test_ignored_ranges_and_oversized_responses_stop(self):
        for status, header, data in [(200, None, b"abcd"), (206, "bytes 1-4/10", b"abcd"),
                                      (206, "bytes 0-3/10", b"abcde")]:
            response = io.BytesIO(data)
            response.status = status
            response.headers = {"Content-Range": header}
            with patch("study_data.urlopen", return_value=response), \
                 RangeFile("https://example.invalid", 10, download_budget(10)) as remote:
                with self.assertRaises(ValueError):
                    remote.read(4)

    def test_rate_limit_stops_other_shard_requests(self):
        budget = download_budget(100)
        error = HTTPError("https://example.invalid", 429, "rate limited", {"Retry-After": "90"}, None)
        with patch("study_data.urlopen", side_effect=error) as request:
            with self.assertRaisesRegex(RuntimeError, "Retry-After: 90"):
                RangeFile("https://example.invalid", 10, budget).read(4)
            with self.assertRaisesRegex(RuntimeError, "previous Parquet request failed"):
                RangeFile("https://example.invalid/other", 10, budget).read(4)
            self.assertEqual(request.call_count, 1)

    def test_audio_plan_counts_each_group_once_and_checks_reference(self):
        clips = [{"audio_id": f"1-2-{i}", "speaker_id": "1", "chapter_id": "2",
                  "reference_text": "A CAT", "dataset": "dev-other"} for i in range(2)]
        index = [{"id": clip["audio_id"], "speaker_id": 1, "chapter_id": 2, "text": "A CAT",
                  "shard_url": "https://example.invalid", "row_group": 0,
                  "audio_group_bytes": 1000} for clip in clips]
        self.assertEqual(estimate_audio_bytes(clips, index), {"row_groups": 1, "compressed_audio_bytes": 1000})
        index[1]["text"] = "OTHER WORDS"
        with self.assertRaisesRegex(ValueError, "differs from selection"):
            selected_groups(clips, index)


if __name__ == "__main__":
    unittest.main()
