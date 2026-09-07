import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class SeedValidationTest(unittest.TestCase):
    script = Path(__file__).with_name("run_seed1_validation.sh")

    def test_dry_run_preserves_fixed_scope(self):
        result = subprocess.run(["bash",str(self.script)],capture_output=True,text=True,check=True)
        commands = result.stdout.strip().splitlines()
        self.assertEqual(len(commands),5)
        self.assertIn("state_dict_8000.ckpt",commands[0])
        self.assertIn("run_checkpoint_crossover",commands[3])
        self.assertIn("--replication",commands[3])
        self.assertIn("run_normalization_sensitivity",commands[4])
        self.assertTrue(all("--execute" not in command for command in commands))

    def test_invalid_argument_and_missing_checkpoint_fail_before_execution(self):
        self.assertEqual(subprocess.run(["bash",str(self.script),"--other"],capture_output=True).returncode,2)
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(["bash",str(self.script),"--execute"],
                env=dict(os.environ,VALIDATION_TRAINING=directory),capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertEqual(result.stdout,"")
            self.assertEqual(list(Path(directory).iterdir()),[])


if __name__ == "__main__":
    unittest.main()
