import unittest
import numpy as np
from .compare_relation_transitions import transition_accounting


class TransitionTests(unittest.TestCase):
    def test_all_nine_states_and_exact_accounting(self):
        y = np.array([1,0,0,0])
        rows, totals = transition_accounting(y,[0,1,0,-1],[0,-1,0,1],
                                            [0,-1,0,1],[0,1,0,-1])
        self.assertEqual(len(rows),9)
        self.assertEqual(sum(r['pairs'] for r in rows),3)
        self.assertAlmostEqual(sum(r['mass'] for r in rows),1)
        self.assertAlmostEqual(sum(r['difference_in_changes_contribution'] for r in rows),
                               totals[3]-totals[2]-totals[1]+totals[0])

    def test_equal_opportunity_can_have_different_retention(self):
        rows,_ = transition_accounting([1,0],[1,0],[0,1],[1,0],[1,0])
        row = next(r for r in rows if r['retweet_early_state']==r['follow_early_state']=='correct')
        self.assertEqual(row['mass'],1)
        self.assertEqual(row['difference_in_changes_contribution'],1)

    def test_advantage_only_does_not_create_shared_retention_effect(self):
        rows,_ = transition_accounting([1,0],[1,0],[0,1],[0,1],[0,1])
        shared = [r for r in rows if r['retweet_early_state']==r['follow_early_state']]
        self.assertEqual(sum(r['mass'] for r in shared),0)
        self.assertEqual(sum(r['difference_in_changes_contribution'] for r in shared),0)

    def test_ties_and_validation(self):
        rows,_ = transition_accounting([1,0],[0,0],[1,0],[0,0],[0,1])
        tied = next(r for r in rows if r['retweet_early_state']==r['follow_early_state']=='tied')
        self.assertEqual(tied['difference_in_changes_contribution'],-1)
        for bad in ([1], [1,float('nan')]):
            with self.assertRaises(ValueError):
                transition_accounting([1,0],bad,[1,0],[1,0],[1,0])


if __name__=='__main__': unittest.main()
