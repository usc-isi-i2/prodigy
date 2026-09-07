"""Control isolation, legacy compatibility, consumed-record and resume tests."""
import copy
import gzip
import json
import random
import tempfile
from pathlib import Path
import unittest

import torch

from data.covid19_twitter import get_covid19_twitter_dataloader
from data.dataloader import BatchSampler, NeighborTask, ParamSampler
from experiments.episode_audit import append_episode_audit, episode_record
from experiments.params import get_params
from experiments.run_shared_graph import validate_configs
from experiments.tests.test_shared_graph_training import tiny_dataset


POLICIES = ("lowest_sorted", "lowest_shuffled", "uniform_sorted", "uniform_shuffled")


class LoggedSampler:
    def __init__(self, sampler):
        self.sampler = sampler
        self.whole_adj = sampler.whole_adj
        self.calls = []

    def random_walk(self, node_idx, direction, generator=None):
        result = self.sampler.random_walk(node_idx, direction, generator=generator)
        self.calls.append((node_idx.tolist(), result.tolist()))
        return result


def make_sampler(policy="lowest_sorted", seed=801):
    graph = tiny_dataset()
    walker = LoggedSampler(graph.neighbor_sampler)
    task = NeighborTask(walker, 24, "inout", strata=[list(range(12)), list(range(12, 24))],
                        confine_to_single_stratum=True, stratum_weighting="balanced", filter_min_degree=True,
                        member_policy=policy, member_sampling_seed=seed)
    return BatchSampler(10, task, ParamSampler(2, 2, 3, 4, 1), seed=51), walker


def make_loader(split, policy):
    dataset = tiny_dataset()
    dataset.nm_validation_neighbor_sampler = dataset.neighbor_sampler
    dataset.nm_test_neighbor_sampler = dataset.neighbor_sampler
    dataset.nm_holdout_neighbor_sampler = dataset.neighbor_sampler
    return get_covid19_twitter_dataloader(
        dataset, split=split, node_split="", batch_size=2, n_way=2, n_shot=3, n_query=4,
        batch_count=2, root="", bert=None, num_workers=0, aug="", aug_test=False,
        split_labels=False, train_cap=None, linear_probe=False, task_name="neighbor_matching",
        neighbor_matching_edge_split=True, neighbor_sampling_episode_source="graph_id",
        neighbor_sampling_episode_source_weighting="balanced", neighbor_matching_member_policy=policy,
        neighbor_matching_member_seed=801)


class MemberPolicyTests(unittest.TestCase):
    def test_legacy_member_draw_and_global_rng_unchanged(self):
        dataset = tiny_dataset()
        task = NeighborTask(
            dataset.neighbor_sampler, 24, "inout", member_policy="lowest_sorted"
        )
        torch.manual_seed(918)
        expected = torch.unique(dataset.neighbor_sampler.random_walk(torch.full((70,), 2), "inout"))[:7].tolist()
        state = torch.get_rng_state().clone()
        torch.manual_seed(918)
        self.assertEqual(task._sample_center_members(2, 7, random.Random(0)), expected)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))

    def test_factorial_treatments_share_anchors_walks_and_retained_sets(self):
        episodes, walks = {}, {}
        for p, policy in enumerate(POLICIES):
            sampler, walker = make_sampler(policy)
            episodes[policy] = []
            for i in range(8):
                torch.rand(100 * (p + 1) + i)  # unrelated model/context randomness
                episodes[policy].append(sampler.sample()[0])
            walks[policy] = walker.calls
        self.assertTrue(all(walks[p] == walks[POLICIES[0]] for p in POLICIES))
        for i in range(8):
            for j in range(2):
                tasks = {p: episodes[p][i][j] for p in POLICIES}
                self.assertTrue(all(list(t) == list(tasks[POLICIES[0]]) for t in tasks.values()))
                for prefix in ("lowest", "uniform"):
                    a, b = tasks[prefix + "_sorted"], tasks[prefix + "_shuffled"]
                    for anchor in a:
                        self.assertEqual(a[anchor], sorted(b[anchor]))
        self.assertNotEqual(episodes["lowest_sorted"], episodes["lowest_shuffled"])
        self.assertNotEqual(episodes["lowest_sorted"], episodes["uniform_sorted"])

    def test_private_rng_resume_and_wrong_policy_rejected(self):
        first, _ = make_sampler("uniform_shuffled")
        first.sample()
        saved = copy.deepcopy(first.state_dict())
        expected = [first.sample() for _ in range(3)]
        second, _ = make_sampler("uniform_shuffled")
        second.load_state_dict(saved)
        actual = [second.sample() for _ in range(3)]
        self.assertEqual(expected, actual)
        wrong, _ = make_sampler("lowest_sorted")
        with self.assertRaisesRegex(ValueError, "different member policy"):
            wrong.load_state_dict(saved)
        old = copy.deepcopy(saved)
        del old["task_sampling_state"]
        with self.assertRaisesRegex(ValueError, "missing dedicated"):
            second.load_state_dict(old)

    def test_eval_policy_is_always_corrected_randomized(self):
        train = make_loader("train", "uniform_shuffled")
        validation = make_loader("val", "uniform_shuffled")
        test = make_loader("test", "uniform_shuffled")
        self.assertEqual(train.batch_sampler.task.member_policy, "uniform_shuffled")
        for loader in (validation, test):
            self.assertEqual(loader.batch_sampler.task.member_policy, "randomized")
            self.assertIsNone(loader.batch_sampler.task.member_generators)

    def test_corrected_member_policy_supports_naive_merged_source_pool(self):
        dataset = tiny_dataset()
        loader = get_covid19_twitter_dataloader(
            dataset, split="train", node_split="", batch_size=1, n_way=2,
            n_shot=3, n_query=4, batch_count=2, root="", bert=None,
            num_workers=0, aug="", aug_test=False, split_labels=False,
            train_cap=None, linear_probe=False, task_name="neighbor_matching",
            neighbor_matching_edge_split=True,
            neighbor_sampling_strata="graph_id_pool",
            neighbor_sampling_source_subset="0,1",
            neighbor_matching_member_policy="uniform_shuffled",
            neighbor_matching_member_seed=801,
        )
        task = loader.batch_sampler.task
        self.assertFalse(task.confine_to_single_stratum)
        self.assertIsNotNone(task.member_generators)
        self.assertEqual(len(task.strata), 1)
        centers = [
            center
            for _ in range(12)
            for center in loader.batch_sampler.sample()[0][0]
        ]
        self.assertTrue(all(0 <= center < 24 for center in centers))
        self.assertTrue(any(center < 12 for center in centers))
        self.assertTrue(any(center >= 12 for center in centers))

    def test_consumed_record_preserves_anchor_and_set_controls(self):
        batches = {p: next(iter(make_loader("train", p))) for p in POLICIES}
        records = {p: episode_record(batch, 1) for p, batch in batches.items()}
        self.assertEqual(len({r["anchor_sha256"] for r in records.values()}), 1)
        for prefix in ("lowest", "uniform"):
            self.assertEqual(records[prefix + "_sorted"]["member_set_sha256"], records[prefix + "_shuffled"]["member_set_sha256"])
        self.assertEqual(records["lowest_sorted"]["query_roles"], [0, 0, 0, 1, 1, 1, 1])
        with tempfile.TemporaryDirectory() as directory:
            append_episode_audit(batches["lowest_sorted"], 1, directory)
            append_episode_audit(batches["lowest_sorted"], 2, directory)
            with gzip.open(Path(directory) / "consumed_episodes.jsonl.gz", "rt") as f:
                self.assertEqual([json.loads(line)["step"] for line in f], [1, 2])

    def test_shared_launcher_allows_only_sampler_not_graph_differences(self):
        base = get_params(["--config", "scripts/experiments/setup/final_core/training.yaml"])
        configs = []
        for policy in POLICIES:
            p = copy.deepcopy(base)
            p.update(neighbor_matching_member_policy=policy, neighbor_matching_member_seed=81, train_episode_audit=True)
            configs.append(p)
        validate_configs(configs)
        configs[-1]["neighbor_matching_walk_hops"] = 2
        with self.assertRaisesRegex(ValueError, "shared-data/protocol"):
            validate_configs(configs)

    def test_unsupported_configs_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "require covid19_twitter"):
            get_params(["--dataset", "twibot20", "--task_name", "nm", "--neighbor_matching_member_seed", "1"])
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            get_params(["--config", "scripts/experiments/setup/final_core/training.yaml", "--neighbor_matching_member_policy", "uniform_sorted"])


if __name__ == "__main__":
    unittest.main()
