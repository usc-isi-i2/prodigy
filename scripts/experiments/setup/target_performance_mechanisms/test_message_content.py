import unittest

import torch

from .test_role_topology import cycle_fixture
from .role_topology import encode, topology_batch
from .message_content import aggregation_audit, message_control
from .verify_member_training import model_digest


class MessageContentTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.model, self.batch = cycle_fixture()
        for layer in self.model.layer_list[0].module_list:
            layer.lin_edge_attr = None

    def test_actual_aggregation_probe_and_scoped_mean(self):
        before = aggregation_audit(self.model)
        with message_control(self.model, self.batch[0], "actual_mean"):
            during = aggregation_audit(self.model)
        self.assertEqual(during[0]["unit_message_probe"], [1., 1., 0.])
        self.assertEqual(before, aggregation_audit(self.model))

    def test_zero_messages_equal_edge_removal(self):
        with torch.no_grad():
            removed, _ = topology_batch(self.batch, "removed", seed=0)
            a, logits_a = encode(self.model, removed)
            with message_control(self.model, self.batch[0], "zero_messages"):
                b, logits_b = encode(self.model, self.batch)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(logits_a, logits_b, rtol=0, atol=0)

    def test_controls_restore_parameters_buffers_and_output(self):
        digest = model_digest(self.model.state_dict())
        with torch.no_grad():
            _, before = encode(self.model, self.batch)
            for condition in ("actual_mean", "no_message_bias", "bias_only", "mean_message"):
                with message_control(self.model, self.batch[0], condition):
                    _, changed = encode(self.model, self.batch)
                    self.assertTrue(torch.isfinite(changed).all())
                _, after = encode(self.model, self.batch)
                torch.testing.assert_close(before, after, rtol=0, atol=0)
                self.assertEqual(digest, model_digest(self.model.state_dict()))

    def test_exception_restores_operator(self):
        before = aggregation_audit(self.model)
        with self.assertRaises(RuntimeError):
            with message_control(self.model, self.batch[0], "actual_mean"):
                raise RuntimeError("test")
        self.assertEqual(before, aggregation_audit(self.model))


if __name__ == "__main__":
    unittest.main()
