import importlib.util
import unittest
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / 'refined_units.py'
spec = importlib.util.spec_from_file_location('refined_units', PATH)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class UnitTests(unittest.TestCase):
    def test_raw_literal_and_unicode_byte_ranges(self):
        source = '// héllo\nvoid target(int x) {\n auto s = R"MSL( } " { // )MSL";\n if (x) {}\n}\nvoid later() {}\n'
        a, b = m.function_range(source, 'void target(')
        self.assertEqual(source.encode()[a:b].decode(), source[source.index('void target'):source.index('void later')])

    def test_signature_in_literal_does_not_match(self):
        source = 'const char* s = "void target(";\nvoid target() { }\n'
        a, b = m.function_range(source, 'void target(')
        self.assertEqual(source.encode()[a:b], b'void target() { }\n')

    def test_duplicate_signature_rejected(self):
        with self.assertRaisesRegex(ValueError, 'exactly once'):
            m.function_range('void target(){}\nvoid target(){}', 'void target(')

    def test_enclosing_condition_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Conditional'):
            m.function_range('#if A\nvoid target() {}\n#endif\n', 'void target(')

    def test_conditional_declaration_group_keeps_all_branches(self):
        raw = "int untouched;\n#if A\nvoid target() {}\n#else\nvoid other() {}\n#endif\nint later;\n"
        a, b = m.function_range(raw, "void target(", include_conditional=True)
        self.assertEqual(raw[a:b], raw[raw.index("#if"):raw.index("int later")])

    def test_internal_condition_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Conditional'):
            m.function_range('void target() {\n#if A\n x();\n#endif\n}\n', 'void target(')

    def test_unterminated_raw_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unterminated'):
            m.function_range('void target(){ auto x=R"a(})b"; }', 'void target(')

    def test_declaration_not_function_rejected(self):
        with self.assertRaisesRegex(ValueError, 'suffix'):
            m.function_range('void target();\nvoid other() {}', 'void target(')

    def test_complete_decorated_method_and_exact_replay(self):
        before = '# π\nclass A:\n    @staticmethod\n    def a():\n        return 1\n\n    def b(self):\n        return 3\n'
        after = before.replace('return 1', 'return 2')
        pair = {'before': {'content': before}, 'after': {'content': after}}
        units = m.python_units('file.py', pair)
        self.assertEqual(len(units), 1)
        u = units[0]
        self.assertEqual(u['kind'], 'complete_function')
        self.assertIn(b'@staticmethod', before.encode()[u['before']['start']:u['before']['end']])
        self.assertNotIn(b'def b', before.encode()[u['before']['start']:u['before']['end']])
        raw = before.encode()
        a, b = u['before']['start'], u['before']['end']
        c, d = u['after']['start'], u['after']['end']
        self.assertEqual(raw[:a] + after.encode()[c:d] + raw[b:], after.encode())
        mutated = after.replace('return 3', 'return 4').encode()
        self.assertNotEqual(raw[:a] + mutated[c:d] + raw[b:], mutated)

    def test_python_new_and_deleted_methods_replay(self):
        before = 'class A:\n    def a(self):\n        return 1\n    def old(self):\n        return 2\n'
        after = 'class A:\n    def a(self):\n        return 1\n    def new(self):\n        return 3\n'
        units = m.python_units('file.py', {'before': {'content': before}, 'after': {'content': after}})
        raw = before.encode()
        for u in reversed(units):
            a, b = u['before']['start'], u['before']['end']
            c, d = u['after']['start'], u['after']['end']
            raw = raw[:a] + after.encode()[c:d] + raw[b:]
        self.assertEqual(raw, after.encode())


if __name__ == '__main__':
    unittest.main()
