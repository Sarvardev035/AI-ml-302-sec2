import os
import tempfile
import unittest

from ml_basics import linear_regression, run_demo, train_xor


class TestMlBasics(unittest.TestCase):
    def test_linear_regression_returns_expected_line(self):
        slope, intercept = linear_regression([1, 2, 3], [3, 5, 7])
        self.assertAlmostEqual(slope, 2.0, places=6)
        self.assertAlmostEqual(intercept, 1.0, places=6)

    def test_xor_training_predictions(self):
        _, predictions = train_xor()
        labels = [1 if p >= 0.5 else 0 for p in predictions]
        self.assertEqual(labels, [0, 1, 1, 0])

    def test_run_demo_creates_plot(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_demo(tmp)
            self.assertTrue(os.path.exists(result["linear_regression"]["plot"]))


if __name__ == "__main__":
    unittest.main()
