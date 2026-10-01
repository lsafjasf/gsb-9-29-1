import unittest
from fractions import Fraction

from reservoir import parse_weight


class TestParseWeight(unittest.TestCase):
    def test_decimal_string_is_exact(self):
        self.assertEqual(parse_weight("0.1"), Fraction(1, 10))
        self.assertEqual(parse_weight("1.5e-3"), Fraction(3, 2000))
        self.assertEqual(parse_weight("0.0000000001"), Fraction(1, 10**10))
        self.assertNotEqual(parse_weight("0.1"), Fraction(1))

    def test_huge_integer_string_and_int(self):
        big = "1234567890123456789012345678901234567890"
        self.assertEqual(parse_weight(big), Fraction(int(big), 1))
        self.assertEqual(parse_weight(10**100), Fraction(10**100, 1))

    def test_other_inputs(self):
        self.assertEqual(parse_weight(Fraction(2, 3)), Fraction(2, 3))
        self.assertEqual(parse_weight("2/3"), Fraction(2, 3))
        self.assertEqual(parse_weight(0), Fraction(0))
        self.assertEqual(parse_weight(0.5), Fraction(1, 2))  # exact binary
        self.assertEqual(parse_weight("  7 "), Fraction(7))

    def test_rejections(self):
        for bad in ("-1", "-0.5", "abc", "", "inf", "nan", "Infinity"):
            with self.assertRaises(ValueError):
                parse_weight(bad)
        with self.assertRaises(ValueError):
            parse_weight(-3)
        with self.assertRaises(TypeError):
            parse_weight(True)
        with self.assertRaises(TypeError):
            parse_weight(None)


if __name__ == "__main__":
    unittest.main()
