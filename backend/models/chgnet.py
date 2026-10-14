"""
CHGNet 模型接口（冻结特征 + MLP 预测头）
支持：band_gap
"""
import numpy as np, torch, os, glob
from pathlib import Path
from models.base import BasePredictor
from pymatgen.core import Structure
sys_path_added = False

PROJECT_ROOT = Path(__file__).resolve().parents[2]

class CHGNetPredictor(BasePredictor):
    name = "CHGNet"

    def __init__(self):
        from chgnet.graph.converter import CrystalGraphConverter
        from chgnet.model.model import CHGNet
        import sys
        global sys_path_added
        if not sys_path_added:
            sys.path.insert(0, str(PROJECT_ROOT))
            sys_path_added = True

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.converter = CrystalGraphConverter()

        print("  加载 CHGNet 特征提取器...")
        self.chgnet = CHGNet.load()
        self.chgnet.eval()
        self.chgnet.to(self.device)

        print("  加载 CHGNet bandgap 预测头...")
        from finetune_chgnet_bandgap import BandGapHead
        cache_dir = PROJECT_ROOT / "data" / "chgnet_finetune"
        model_files = sorted(glob.glob(os.path.join(cache_dir, "model_*.pt")),
                             key=lambda x: -int(x.split("_")[-1].split(".")[0]))
        if not model_files:
            raise FileNotFoundError("未找到 CHGNet MLP 模型，请先运行 finetune_chgnet_bandgap.py")
        ckpt = torch.load(model_files[0], map_location=self.device, weights_only=False)
        self.mean = ckpt["mean"]
        self.std  = ckpt["std"]
        self.head = BandGapHead(64, [128, 64, 32]).to(self.device)
        self.head.load_state_dict(ckpt["model_state_dict"])
        self.head.eval()

    def _graph_to_device(self, graph):
        for attr in dir(graph):
            if attr.startswith("_"): continue
            try:
                val = getattr(graph, attr)
                if isinstance(val, torch.Tensor):
                    setattr(graph, attr, val.to(self.device))
            except: pass
        return graph

    def predict(self, structure: Structure, property_name: str) -> dict:
        if property_name != "band_gap":
            raise ValueError("CHGNet 仅支持 band_gap 预测")
        graph = self.converter(structure)
        self._graph_to_device(graph)
        with torch.no_grad():
            result = self.chgnet([graph], task="e", return_crystal_feas=True)
            pred = self.head(result["crystal_fea"]).item()
        pred = pred * self.std + self.mean
        return {"value": round(pred, 4), "unit": "eV"}

    def is_available(self) -> bool:
        return True
