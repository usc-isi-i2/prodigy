"""Guard against silently overridden components in the combined recipe."""
import unittest
from scripts.experiments.setup.nm_interventions_overnight.select_combination import compatible_arms


class CompatibilityTests(unittest.TestCase):
    def test_sampler_precedence_conflicts(self):
        for a,b in [('composition','schedule'),('exposure','schedule'),
                    ('composition','negatives'),('eligibility','positives'),
                    ('centers','coverage')]:
            scores={a:dict(eligible=True,mean_delta=.03),b:dict(eligible=True,mean_delta=.02)}
            chosen,excluded=compatible_arms(scores)
            self.assertEqual(chosen,[a])
            self.assertIn(b,excluded)

    def test_compatible_effects_are_retained(self):
        scores={a:dict(eligible=True,mean_delta=.02) for a in
                ['exposure','composition','alignment','sharing','optimization','capacity']}
        chosen,excluded=compatible_arms(scores)
        self.assertEqual(set(chosen),set(scores))
        self.assertFalse(excluded)

    def test_adaptation_requires_gain_over_centers(self):
        scores={'centers':dict(eligible=False,mean_delta=.02),
                'region_adaptive':dict(eligible=True,mean_delta=.019)}
        chosen,excluded=compatible_arms(scores)
        self.assertFalse(chosen)
        self.assertIn('region_adaptive',excluded)

if __name__=='__main__':unittest.main()
