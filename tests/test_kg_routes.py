from pathlib import Path
import pickle
import unittest


ROOT = Path(__file__).resolve().parents[1]


class RuntimeArtifactTests(unittest.TestCase):
    def test_frozen_graph_is_loadable(self):
        path = ROOT / "data" / "knowledge_graph" / "sci_tool_kg.pkl"
        with path.open("rb") as handle:
            graph = pickle.load(handle)
        self.assertEqual(graph.number_of_nodes(), 39)
        self.assertEqual(graph.number_of_edges(), 142)

    def test_required_runtime_artifacts_exist(self):
        required = [
            ROOT / "data" / "routing_table" / "mae_matrix.json",
            ROOT / "data" / "routing_table" / "result.json",
            ROOT / "data" / "reference" / "element_reference_energies.json",
            ROOT / "data" / "chgnet_finetune" / "model_108056.pt",
        ]
        self.assertTrue(all(path.is_file() for path in required))


if __name__ == "__main__":
    unittest.main()
