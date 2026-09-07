import copy
import unittest

from .verify import compare_payloads


def row(source, value):
    return dict(source_ids=[source], anchor_ids=[value], member_ids=[value],
        query_roles=[False, True], context_node_counts=[2], anchor_sha256=str(value),
        member_order_sha256=str(value), member_set_sha256=str(value),
        context_node_order_sha256=str(value), context_edge_sha256=str(value))


class PayloadTest(unittest.TestCase):
    def test_source_prefixes_allow_shorter_smoke_schedule(self):
        reference = [row(0, 1), row(0, 2), row(1, 3), row(1, 4)]
        self.assertEqual(compare_payloads([row(0, 1), row(1, 3)], reference), {"0": 1, "1": 1})

    def test_changed_roles_or_missing_source_fail(self):
        reference = [row(0, 1), row(1, 3)]
        changed = copy.deepcopy(reference)
        changed[0]["query_roles"] = [True, False]
        with self.assertRaises(ValueError):
            compare_payloads(changed, reference)
        with self.assertRaises(ValueError):
            compare_payloads(reference[:1], reference)

    def test_per_source_order_is_not_ignored(self):
        reference = [row(0, 1), row(0, 2)]
        with self.assertRaises(ValueError):
            compare_payloads(reference[::-1], reference)


if __name__ == "__main__":
    unittest.main()
