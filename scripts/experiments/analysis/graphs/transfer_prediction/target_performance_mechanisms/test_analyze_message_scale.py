import unittest
import pandas as pd

from .analyze_message_scale import pair_scores


class ScaleAnalysisTest(unittest.TestCase):
    def test_own_baselines_and_no_duplicate_conditions(self):
        rows=[]
        for model,base in [('a',.8),('b',.6)]:
            for condition,change in [('scale_1',0.),('scale_0',.05)]:
                rows.append(dict(target='t',stream='s',model_id=model,condition=condition,
                                 roc_auc=base+change,accuracy=base+change,f1=base+change,nll=1-base-change))
        cells=pd.DataFrame(rows[::-1]);paired=pair_scores(cells)
        self.assertTrue(((paired[paired.condition=='scale_0'].delta_roc_auc-.05).abs()<1e-12).all())
        with self.assertRaises(ValueError):pair_scores(pd.concat([cells,cells.iloc[:1]]))
        with self.assertRaises(ValueError):pair_scores(cells[cells.condition!='scale_1'])


if __name__=='__main__':unittest.main()
