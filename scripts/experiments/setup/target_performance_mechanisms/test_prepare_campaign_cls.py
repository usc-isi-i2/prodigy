import unittest

from .prepare_campaign_cls import ALLOWED_FLAGS, EXCLUDED_FLAGS, FORWARD, select_records


def fixture():
    flags = [[]] + [[f] for f in sorted(ALLOWED_FLAGS)] + [["aux_reconstruction"]] + [[f] for f in sorted(EXCLUDED_FLAGS)]
    return [{"model_id": f"model_{i}",
             "params": dict(FORWARD, seed=0, batch_size=1, learning_rate=.001,
                            n_way=30, n_shots=3, n_query=4, edge_view="static_train"),
             "selection": {"sources": list("abcdefgh"), "flags": f, "status": "complete",
                           "exposure": {"twibot20": 0}, "training_steps": 6000 if i == 0 else 10000},
             "runtime": {"result": {"status": "complete"}}} for i, f in enumerate(flags)]


class CampaignManifestTests(unittest.TestCase):
    def test_complete_subset_and_common_checkpoint(self):
        included, excluded, step = select_records(fixture())
        self.assertEqual((len(included), len(excluded), step), (15, 3, 6000))

    def test_forward_unknown_holdout_and_budget_fail_closed(self):
        for field, value in (("emb_dim", 128), ("batch_size", 4), ("n_way", 2)):
            records = fixture(); records[0]["params"][field] = value
            with self.assertRaises(ValueError):
                select_records(records)
        records = fixture(); records[0]["selection"]["flags"] = ["unknown"]
        with self.assertRaises(ValueError):
            select_records(records)
        records = fixture(); records[0]["selection"]["exposure"]["twibot20"] = 1
        with self.assertRaises(ValueError):
            select_records(records)
        records = fixture(); records[0]["selection"]["training_steps"] = 4000
        with self.assertRaises(ValueError):
            select_records(records)
        with self.assertRaises(ValueError):
            select_records(fixture()[:-1])


if __name__ == "__main__":
    unittest.main()
