import json
from pathlib import Path
import tempfile
import unittest
from .run import commands
from .eval_plan import prepare, TARGETS


class EvalPlanTest(unittest.TestCase):
    def fixture(self,root):
        plan=commands(root,root,0,2500,0)
        (root/'DONE.json').write_text(json.dumps(dict(arms=8,steps=2500,smoke=False)))
        (root/'plan.json').write_text(json.dumps(plan))
        for e in plan:
            p=root/'state'/(e['name']+'_isolation_v1')/'checkpoint'/'state_dict_2500.ckpt'
            p.parent.mkdir(parents=True);p.touch()
        return plan

    def test_complete_shared_panel(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture(root)
            manifest,cmds=prepare(root,root/'eval',0)
            self.assertEqual(len(manifest.splitlines()),9)
            self.assertEqual(len(cmds),10)
            self.assertEqual({c[c.index('--datasets')+1] for c in cmds},set(TARGETS))
            self.assertEqual({c[c.index('--eval-episode-seed-offset')+1] for c in cmds},{'0','100003'})
            self.assertFalse((root/'eval').exists())

    def test_reject_smoke_and_missing_arm(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);plan=self.fixture(root)
            (root/'plan.json').write_text(json.dumps(plan[:-1]))
            with self.assertRaises(ValueError):prepare(root,root/'eval',0)
            (root/'plan.json').write_text(json.dumps(plan))
            (root/'DONE.json').write_text(json.dumps(dict(arms=8,steps=20,smoke=True)))
            with self.assertRaises(ValueError):prepare(root,root/'eval',0)


if __name__=='__main__':unittest.main()
