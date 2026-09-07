import unittest
import torch
from .role_centering import predict


class CenteringTests(unittest.TestCase):
    def test_separate_offsets_and_input_preservation(self):
        torch.manual_seed(8)
        s, q = torch.randn(6, 8, dtype=torch.float64), torch.randn(8, 8, dtype=torch.float64)
        y = torch.eye(2, dtype=torch.float64).repeat_interleave(3, 0)
        before_s, before_q = s.clone(), q.clone()
        expected = predict(s, y, q, "separate")
        torch.testing.assert_close(expected, predict(s + 17, y, q - 12, "separate"))
        torch.testing.assert_close(s, before_s)
        torch.testing.assert_close(q, before_q)
        torch.testing.assert_close(predict(s, y, q, "support"), predict(s + 3, y, q + 3, "support"))

    def test_labels_and_query_order(self):
        s = torch.eye(3).repeat_interleave(3, 0)
        y = s.clone()
        q = torch.eye(3)
        for center in ("none", "support", "separate"):
            scores = predict(s, y, q, center)
            torch.testing.assert_close(scores.argmax(1), torch.arange(3))
            torch.testing.assert_close(predict(s, y[:, [2, 0, 1]], q, center), scores[:, [2, 0, 1]])
            torch.testing.assert_close(predict(s, y, q[[2, 0, 1]], center), scores[[2, 0, 1]])


if __name__ == "__main__":
    unittest.main()
