import io
import unittest

from blurface.app import _configure_console_stream


class ConsoleCompatibilityTests(unittest.TestCase):
    def test_unencodable_paths_are_escaped_on_legacy_windows_console(self):
        raw = io.BytesIO()
        stream = io.TextIOWrapper(raw, encoding="cp1252")
        _configure_console_stream(stream)
        print("C:/视频/output.mp4", file=stream)
        stream.flush()
        self.assertIn(b"\\u", raw.getvalue())


if __name__ == "__main__":
    unittest.main()
