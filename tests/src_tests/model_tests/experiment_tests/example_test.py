import unittest


def function():
    print("test_function")
    return 3


class TestExample(unittest.TestCase):
    def test_function(self):
        result = function()
        self.assertEqual(result, 3)


if __name__ == '__main__':
    unittest.main()
