import unittest
from .audit_support_cue import describe


class CueTests(unittest.TestCase):
    def test_polarities(self):
        for y in ([0,0,1,1],[1,1,0,0]):
            r=describe(y,[0,0,1,1],y,[0,0,1,1])
            self.assertEqual(r['support_exception_rate'],0)
            self.assertEqual(r['query_cue_auc'],1)

    def test_query_labels_do_not_choose_polarity(self):
        a=describe([0,0,1,1],[0,0,1,1],[0,1],[0,1])
        b=describe([0,0,1,1],[0,0,1,1],[1,0],[0,1])
        self.assertEqual(a['support_polarity'],b['support_polarity'])
        self.assertEqual(b['query_cue_auc'],0)

    def test_tie(self):
        r=describe([0,0,1,1],[0,1,0,1],[0,1],[0,1])
        self.assertEqual(r['support_exception_rate'],.5)
        self.assertEqual(r['query_cue_auc'],.5)


if __name__=='__main__': unittest.main()
