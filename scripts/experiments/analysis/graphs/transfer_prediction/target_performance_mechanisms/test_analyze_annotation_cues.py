import copy
import json
from pathlib import Path
import unittest

from .analyze_annotation_cues import validate_and_summarize


class CompletedCueAudit(unittest.TestCase):
    def setUp(self):
        root=Path(__file__).parent/"data/annotation_cues_20260907"
        self.values=[json.loads((root/(n+".json")).read_text()) for n in ("protocol","DONE","metrics","receipts","provenance","support_counts")]

    def test_complete(self):
        result=validate_and_summarize(*self.values)
        self.assertEqual(len(result["subset_counts"]),32)
        self.assertEqual(len(result["paired"]),1152)

    def test_missing_cell(self):
        self.values[2].pop()
        with self.assertRaises(ValueError): validate_and_summarize(*self.values)

    def test_changed_stratum(self):
        self.values[2][0]["unique_queries"]-=1
        with self.assertRaises(ValueError): validate_and_summarize(*self.values)

    def test_bad_parity(self):
        self.values[3][0]["metric_max_abs_error"]=.01
        with self.assertRaises(ValueError): validate_and_summarize(*self.values)

    def test_unsupported_metric(self):
        row=next(r for r in self.values[2] if r["queries"]==0)
        row["roc_auc"]=.5
        with self.assertRaises(ValueError): validate_and_summarize(*self.values)

    def test_failed_provenance(self):
        self.values[4]["unique_complete_row_matches"]-=1
        with self.assertRaises(ValueError): validate_and_summarize(*self.values)


if __name__=="__main__": unittest.main()
