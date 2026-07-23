"""
第一阶段：查表路由
读取 mae_matrix.json，按晶系+性质返回最优模型
"""
import json
from pathlib import Path


class TableRouter:
    def __init__(self, matrix_path: str):
        self.matrix_path = Path(matrix_path)
        self.matrix = self._load_matrix()

    def _load_matrix(self) -> dict:
        if not self.matrix_path.exists():
            return {}
        with open(self.matrix_path) as f:
            return json.load(f)

    def route(self, crystal_system: str, prop: str) -> dict:
        """
        返回格式:
        {
            "best_model": "ALIGNN",
            "mae": 0.31,
            "sample_count": 850,
            "all_models": {"ALIGNN": 0.31, "MEGNet": 0.45, ...}
        }
        """
        key = f"{crystal_system}_{prop}"
        entry = self.matrix.get(key)

        if not entry:
            return {"best_model": None, "mae": None, "sample_count": 0, "all_models": {}}

        all_models    = entry.get("models", {})
        sample_count  = entry.get("sample_count", 0)

        if not all_models:
            return {"best_model": None, "mae": None, "sample_count": sample_count, "all_models": {}}

        best_model = min(all_models, key=all_models.get)
        return {
            "best_model":   best_model,
            "mae":          all_models[best_model],
            "sample_count": sample_count,
            "all_models":   all_models,
        }
