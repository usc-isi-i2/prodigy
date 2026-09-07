import unittest
from pathlib import Path
from .run import commands, MODES


class PlanTest(unittest.TestCase):
    def test_gradient_arms_share_every_training_argument_except_mode_and_name(self):
        plan=commands(Path('/repo'),Path('/output'),0,2500,0)
        self.assertEqual(len(plan),8)
        for schedule in ('blocked','interleaved'):
            arms=[x for x in plan if x['arm']['schedule']==schedule]
            self.assertEqual([x['mode'] for x in arms],list(MODES))
            signatures=[]
            for arm in arms:
                cmd=arm['command'].copy()
                self.assertEqual(cmd[cmd.index('--checkpoint_steps')+1],'0,2500')
                for flag in ('--prefix','--encoder_solver_objective'):
                    cmd[cmd.index(flag)+1]='VARYING'
                signatures.append(cmd)
                self.assertEqual(arm['arm']['source_counts'],(625,625,625,625))
            self.assertTrue(all(s==signatures[0] for s in signatures))

    def test_gpu_and_budget_boundaries(self):
        for gpu,steps in [(4,20),(0,19)]:
            with self.assertRaises(ValueError):commands(Path('/repo'),Path('/output'),gpu,steps,0)


if __name__=='__main__':unittest.main()
