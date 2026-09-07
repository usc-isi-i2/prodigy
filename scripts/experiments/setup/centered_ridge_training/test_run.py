import unittest
from pathlib import Path
from .run import commands,MODE
from scripts.experiments.setup.encoder_solver_isolation.run import commands as old_commands


class PlanTest(unittest.TestCase):
    def test_exact_old_protocol_except_mode_name(self):
        for steps in (20,2500):
            new=commands(Path('/repo'),Path('/out'),3,steps)
            old=[x for x in old_commands(Path('/repo'),Path('/out'),3,steps,0) if x['mode']=='ridge_only']
            self.assertEqual(len(new),2)
            for a,b in zip(new,old):
                self.assertEqual(a['arm'],b['arm'])
                x,y=a['command'].copy(),b['command'].copy()
                for flag in ('--prefix','--encoder_solver_objective'):
                    x[x.index(flag)+1]=y[y.index(flag)+1]='vary'
                self.assertEqual(x,y)
                self.assertEqual(a['mode'],MODE)
        with self.assertRaises(ValueError):commands(Path('/r'),Path('/o'),3,2500,1)


if __name__=='__main__':unittest.main()
