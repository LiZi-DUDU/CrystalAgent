"""
ALIGNN 模型接口
支持：形成能、带隙
预加载模型避免每次重复加载
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

    def __init__(self):
        from alignn.pretrained import get_figshare_model
        self._models = {}
        for prop, model_name in MODEL_MAP.items():
            print(f'加载 ALIGNN {prop} 模型...')
            self._models[prop] = get_figshare_model(model_name).to('cpu')
        print('ALIGNN 所有模型加载完成')

    def predict(self, structure: Structure, property_name: str) -> dict:
        if property_name not in MODEL_MAP:
            raise ValueError(f"ALIGNN 不支持预测 {property_name}")

        import torch
        from alignn.graphs import Graph
        from jarvis.core.atoms import Atoms

        atoms = JarvisAtomsAdaptor.get_atoms(structure)
        model = self._models[property_name]

        # 直接用已加载的模型推理，不重新加载
        jarvis_atoms = Atoms(
            lattice_mat=atoms.lattice_mat,
            coords=atoms.coords,
            elements=atoms.elements,
            cartesian=False,
        )
        g, lg = Graph.atom_dgl_multigraph(jarvis_atoms)
        lat = torch.tensor(atoms.lattice_mat, dtype=torch.float32)
        with torch.no_grad():
            out = model([g, lg, lat])
        value = out.cpu().numpy().flatten()[0]

        return {
            "value": round(float(value), 4),
            "unit":  UNIT_MAP[property_name],
        }

    def is_available(self) -> bool:
        return len(self._models) > 0