"""
MEGNet 模型接口（通过 matgl 包调用）
支持：形成能、带隙
"""
from models.base import BasePredictor
from pymatgen.core import Structure


MODEL_MAP = {
    "formation_energy": "MEGNet-Eform-MP-2018.6.1",
    "band_gap":         "MEGNet-BandGap-mfi-MP-2019.4.1",
}

UNIT_MAP = {
    "formation_energy": "eV/atom",
    "band_gap":         "eV",
}


class MEGNetPredictor(BasePredictor):
    name = "MEGNet"

    def __init__(self):
        import matgl
        self.models = {}
        for prop, model_name in MODEL_MAP.items():
            try:
                self.models[prop] = matgl.load_model(model_name)
            except Exception as e:
                print(f"MEGNet 加载 {model_name} 失败: {e}")

    def predict(self, structure: Structure, property_name: str) -> dict:
        if property_name not in MODEL_MAP:
            raise ValueError(f"MEGNet 不支持预测 {property_name}")

        model = self.models.get(property_name)
        if model is None:
            raise RuntimeError(f"MEGNet {property_name} 模型未加载")

        import torch
        with torch.no_grad():
            if property_name == "band_gap":
                result = model.predict_structure(structure, state_attr=torch.tensor([0]))
            else:
                result = model.predict_structure(structure)

        return {
            "value": round(float(result), 4),
            "unit":  UNIT_MAP[property_name],
        }

    def is_available(self) -> bool:
        return len(self.models) > 0