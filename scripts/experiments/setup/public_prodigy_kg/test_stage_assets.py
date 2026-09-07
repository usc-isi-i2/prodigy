import io
import stat
import unittest
import zipfile

from .stage_assets import EXTRA_REQUIRED, FEATURE_FILE, checked_members


class ArchiveSafetyTests(unittest.TestCase):
    def archive(self, extra=None):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            for name in ("Wiki/graph.pt", "Wiki/Wiki_adj.pt", f"Wiki/{FEATURE_FILE}",
                         *(f"Wiki/{path}" for path in EXTRA_REQUIRED["Wiki"])):
                archive.writestr(name, b"fixture, not serialized model data")
            if extra is not None:
                archive.writestr(extra, b"unexpected")
        buffer.seek(0)
        return zipfile.ZipFile(buffer)

    def test_expected_members(self):
        with self.archive() as archive:
            self.assertEqual(len(checked_members(archive, "Wiki")), 6)

    def test_traversal_absolute_windows_and_other_root(self):
        for name in ("Wiki/../../bad", "/tmp/bad", "Wiki\\bad", "Wiki/C:bad", "Other/bad"):
            with self.subTest(name=name), self.archive(name) as archive:
                with self.assertRaises(ValueError):
                    checked_members(archive, "Wiki")

    def test_symlink(self):
        item = zipfile.ZipInfo("Wiki/link")
        item.create_system = 3
        item.external_attr = (stat.S_IFLNK | 0o777) << 16
        with self.archive(item) as archive:
            with self.assertRaises(ValueError):
                checked_members(archive, "Wiki")

    def test_mac_metadata_skipped(self):
        with self.archive("__MACOSX/Wiki/._graph.pt") as archive:
            self.assertEqual(len(checked_members(archive, "Wiki")), 6)

    def test_wrong_dataset_missing_assets(self):
        with self.archive() as archive:
            with self.assertRaises(ValueError):
                checked_members(archive, "FB15K-237")


if __name__ == "__main__":
    unittest.main()
