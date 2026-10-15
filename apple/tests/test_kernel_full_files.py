import copy
import json
import unittest

from exl3_mlx import kernel_full_files as f


class FullFilesTests(unittest.TestCase):
    def setUp(self):
        self.example = {
            "format": "kernel-full-files-v1",
            "rationale": "Retain the baseline pending measurements.",
            "files": {
                "kernel.metal": "// fixture, not executable proof\n",
                "dispatch.json": '{"threads_per_row":1}\n',
            },
        }

    def encoded(self, value):
        return json.dumps(value).encode()

    def test_exact_baseline_images_and_noop_allowed(self):
        files = f.parse(self.encoded(self.example))
        for name, data in files.items():
            self.assertEqual(data, self.example["files"][name].encode())

    def test_only_two_exact_paths(self):
        for key in ("../kernel.metal", "/tmp/kernel.metal", "extra.metal"):
            x = copy.deepcopy(self.example)
            x["files"][key] = "x\n"
            with self.assertRaises(ValueError):
                f.parse(self.encoded(x))
        del self.example["files"]["dispatch.json"]
        with self.assertRaises(ValueError):
            f.parse(self.encoded(self.example))

    def test_duplicate_keys_and_nonfinite(self):
        for raw in (b'{"format":1,"format":2}', b'{"x":NaN}'):
            with self.assertRaises(ValueError):
                f.parse(raw)
        self.example["files"]["dispatch.json"] = '{"threads_per_row":1,"threads_per_row":32}\n'
        with self.assertRaises(ValueError):
            f.parse(self.encoded(self.example))

    def test_dispatch_scalar_and_schema(self):
        for value in (True, 1.0, 0, 2, 64, "32"):
            self.example["files"]["dispatch.json"] = json.dumps({"threads_per_row": value}) + "\n"
            with self.assertRaises(ValueError):
                f.parse(self.encoded(self.example))
        self.example["files"]["dispatch.json"] = '{"threads_per_row":32}\n'
        self.assertEqual(
            f.parse(self.encoded(self.example))["dispatch.json"], b'{"threads_per_row":32}\n'
        )

    def test_bounds_and_encoding(self):
        for value in ("x" * (f.MAX_KERNEL + 1), "x\x00\n", "x\r\n", "x", "\ud800\n"):
            self.example["files"]["kernel.metal"] = value
            with self.assertRaises((ValueError, UnicodeError)):
                f.parse(self.encoded(self.example))
        for raw in (b"\xff", b" " * (f.MAX_RESPONSE + 1)):
            with self.assertRaises((ValueError, UnicodeError)):
                f.parse(raw)

    def test_no_shell_wrappers_markdown_or_trailing_objects(self):
        raw = self.encoded(self.example)
        for value in (
            b"```json\n" + raw + b"\n```",
            b"cat <<EOF\n" + raw + b"\nEOF",
            raw + b"{}",
            raw + b" prose",
        ):
            with self.assertRaises(ValueError):
                f.parse(value)

    def test_extra_response_fields_and_empty_rationale(self):
        self.example["command"] = "echo should never execute"
        with self.assertRaises(ValueError):
            f.parse(self.encoded(self.example))
        del self.example["command"]
        self.example["rationale"] = ""
        with self.assertRaises(ValueError):
            f.parse(self.encoded(self.example))

    def test_parser_does_not_claim_source_safety(self):
        self.example["files"]["kernel.metal"] = (
            "not valid Metal; requires later compiler and source review\n"
        )
        self.assertEqual(
            f.parse(self.encoded(self.example))["kernel.metal"],
            self.example["files"]["kernel.metal"].encode(),
        )

    def test_excessive_nesting_is_a_format_failure(self):
        with self.assertRaises(ValueError):
            f.parse(b"[" * 2000 + b"0" + b"]" * 2000)


if __name__ == "__main__":
    unittest.main()
