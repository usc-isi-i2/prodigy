import unittest
import numpy as np
import pandas as pd

from .annotation_cues import cue_flags, join_original_profiles, subset_masks, PUBLISHED_CUES


class CueContracts(unittest.TestCase):
    def test_published_membership(self):
        self.assertEqual(len(PUBLISHED_CUES), 29)
        for word in PUBLISHED_CUES:
            self.assertEqual(cue_flags("#" + word.swapcase(), ""), (True, True))

    def test_markup_and_cleaned_aliases_are_distinct(self):
        self.assertEqual(cue_flags("Trump #gardening", "Trump"), (False, True))
        self.assertEqual(cue_flags("#Biden2020", "Biden"), (True, True))
        self.assertEqual(cue_flags("", "WWGWGA"), (False, True))
        self.assertEqual(cue_flags("#notMAGA", "unresisting"), (False, False))
        self.assertEqual(cue_flags("＃MAGA", ""), (True, True))

    def test_complete_row_join_preserves_order(self):
        full = pd.DataFrame({"profile": ["same", "same", "third"], "label_conservative": [0, 1, 0],
                             "n_posts": [3, 2, 1], "raw_profile": ["left", "right", "other"]})
        small = full.drop(columns="raw_profile").iloc[[1, 0]].reset_index(drop=True)
        self.assertEqual(join_original_profiles(small, full), ["right", "left"])
        with self.assertRaises(ValueError): join_original_profiles(small, pd.concat([full, full.iloc[:1]]))
        with self.assertRaises(ValueError): join_original_profiles(small, full.iloc[1:])

    def test_partitions_and_containment(self):
        m = subset_masks([False, False, True], [False, True, True], [False, True, False])
        np.testing.assert_array_equal(m["query_center_absent"] | m["query_center_present"], m["all"])
        np.testing.assert_array_equal(m["all_input_absent"], [True, False, False])
        with self.assertRaises(ValueError): subset_masks([True], [False], [False])

    def test_global_labels_and_repeated_account_scoring(self):
        import torch
        from .run_annotation_cues import metrics
        labels = {"local_y": torch.tensor([0, 1, 1, 0]), "mapping": torch.tensor([[0,1],[0,1],[1,0],[1,0]]),
                  "episode_ids": torch.tensor([0,0,1,1]), "use_global": True}
        logits = torch.tensor([[2.,0.],[0.,2.],[0.,2.],[2.,0.]])
        ids = torch.tensor([10,20,10,20])
        r = metrics(logits, labels, np.ones(4, bool), ids)
        self.assertEqual((r["queries"],r["unique_queries"],r["roc_auc"],r["unique_account_auc"]), (4,2,1.,1.))
        empty = metrics(logits,labels,np.zeros(4,bool),ids)
        self.assertIsNone(empty["roc_auc"])
        self.assertEqual(empty["queries"],0)
        labels["local_y"][2] = 0
        with self.assertRaises(ValueError): metrics(logits,labels,np.ones(4,bool),ids)


if __name__ == "__main__": unittest.main()
