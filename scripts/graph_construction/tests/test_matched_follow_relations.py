import unittest
from scripts.graph_construction.matched_follow_relations import canonical_pair, verify_sample


class MatchedFollowTests(unittest.TestCase):
    def test_direction(self):
        self.assertEqual(canonical_pair("a", "friend", "b"), ("a", "b"))
        self.assertEqual(canonical_pair("a", "follow", "b"), ("b", "a"))
        with self.assertRaises(ValueError): canonical_pair("a", "post", "b")

    def test_public_original_contract(self):
        sample=[{"ID":"1", "neighbor":{"following":["2"], "follower":["3"]}}]
        observed={("u1","friend"):{"u2"}, ("u1","follow"):{"u3"}}
        self.assertEqual(verify_sample(sample,observed), {"friend":1,"follow":1})
        with self.assertRaises(ValueError):
            verify_sample(sample,{("u1","friend"):{"u3"},("u1","follow"):{"u2"}})

    def test_empty_evidence_rejected(self):
        with self.assertRaises(ValueError): verify_sample([{"ID":"1", "neighbor":None}], {})


if __name__ == "__main__": unittest.main()
