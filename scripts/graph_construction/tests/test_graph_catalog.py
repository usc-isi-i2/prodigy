import json
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[3]
CATALOG_PATH = REPO_ROOT / "config" / "graph_catalog.json"


class GraphCatalogTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
        cls.graphs = cls.catalog["graphs"]

    def test_current_data_root(self):
        self.assertEqual(self.catalog["data_root"], "/dataMeR1/phil/data")

    def test_names_and_keys_are_unique(self):
        canonical_names = [graph["canonical_name"] for graph in self.graphs]
        dataset_keys = [graph["dataset_key"] for graph in self.graphs]
        self.assertEqual(len(canonical_names), len(set(canonical_names)))
        self.assertEqual(len(dataset_keys), len(set(dataset_keys)))

    def test_canonical_source_names(self):
        source_names = [
            graph["canonical_name"] for graph in self.graphs if graph["kind"] == "source"
        ]
        self.assertEqual(
            source_names,
            [
                "ukraine",
                "covid",
                "midterm",
                "covid-political",
                "ukraine-suspended",
                "election2020-political",
                "twibot20",
                "hongkong",
            ],
        )

    def test_paths_are_relative_graph_artifacts(self):
        for graph in self.graphs:
            path = Path(graph["relative_path"])
            self.assertFalse(path.is_absolute(), graph["dataset_key"])
            self.assertEqual(path.parent.name, "graphs", graph["dataset_key"])

    def test_inventory_fields_are_present(self):
        for graph in self.graphs:
            with self.subTest(graph=graph["dataset_key"]):
                self.assertGreater(graph["artifact_size_bytes"], 0)
                self.assertGreater(graph["artifact_size_gb"], 0)
                self.assertGreater(graph["statistics"]["nodes"], 0)
                self.assertGreater(graph["statistics"]["edges"], 0)
                self.assertTrue(graph["tasks"]["supported"])

    def test_source_graphs_document_inputs_and_labels(self):
        for graph in self.graphs:
            if graph["kind"] != "source":
                continue
            with self.subTest(graph=graph["dataset_key"]):
                self.assertIn("metadata_path", graph)
                self.assertTrue(graph["source_data"])
                self.assertIn("features", graph)
                self.assertIn("labels", graph)

    def test_merged_sources_exist(self):
        canonical_names = {graph["canonical_name"] for graph in self.graphs}
        for graph in self.graphs:
            for source in graph.get("sources", []):
                self.assertIn(source, canonical_names, graph["dataset_key"])


if __name__ == "__main__":
    unittest.main()
