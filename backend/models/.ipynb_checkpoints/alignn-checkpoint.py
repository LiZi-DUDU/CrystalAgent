"""
ALIGNN 模型接口
支持：形成能、带隙
"""
from models.base import BasePredictor
from pymatgen.core import Structure
from pymatgen.io.jarvis import JarvisAtomsAdaptor


MODEL_MAP = {
    "formation_energy": "jv_formation_energy_peratom_alignn",
    "band_gap":         "jv_optb88vdw_bandgap_alignn",
}

UNIT_MAP = {
    "formation_energy": "eV/atom",
    "band_gap":         "eV",
}


class ALIGNNPredictor(BasePredictor):
    name = "ALIGNN"

    def predict(self, structure: Structure, property_name: str) -> dict:
        if property_name not in MODEL_MAP:
            raise ValueError(f"ALIGNN 不支持预测 {property_name}")

        from alignn.pretrained import get_prediction

        atoms = JarvisAtomsAdaptor.get_atoms(structure)
        result = get_prediction(
            model_name=MODEL_MAP[property_name],
            atoms=atoms,
        )

        value = result[0] if isinstance(result, list) else float(result)
        return {
            "value": round(float(value), 4),
            "unit":  UNIT_MAP[property_name],
        }

    def is_available(self) -> bool:
        return True