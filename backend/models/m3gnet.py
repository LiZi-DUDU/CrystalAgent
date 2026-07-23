"""
M3GNet 模型接口（通过 matgl 包调用）
支持：形成能
"""
from models.base import BasePredictor
from pymatgen.core import Structure


class M3GNetPredictor(BasePredictor):
    name = "M3GNet"

    def __init__(self):
        import matgl
        try:
            self.model = matgl.load_model("M3GNet-Eform-MP-2018.6.1")
        except Exception as e:
            print(f"M3GNet 加载失败: {e}")
            self.model = None

    def predict(self, structure: Structure, property_name: str) -> dict:
        if property_name != "formation_energy":
            raise ValueError(f"M3GNet 不支持预测 {property_name}")

        import torch
        with torch.no_grad():
            result = self.model.predict_structure(structure)

        return {
            "value": round(float(result), 4),
            "unit":  "eV/atom",
        }

    def is_available(self) -> bool:
        return self.model is not None