"""
CGCNN 模型接口
支持：形成能、带隙、体模量、剪切模量
"""
import sys
import os
import torch
import numpy as np
from pathlib import Path
from pymatgen.core import Structure
from models.base import BasePredictor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CGCNN_DIR = PROJECT_ROOT / 'cgcnn'
if str(CGCNN_DIR) not in sys.path:
    sys.path.insert(0, str(CGCNN_DIR))

from cgcnn.model import CrystalGraphConvNet
from cgcnn.data import CIFData, collate_pool

PRETRAINED_DIR = CGCNN_DIR / 'pre_trained'

MODEL_FILES = {
    'formation_energy': 'formation-energy-per-atom.pth.tar',
    'band_gap':         'band-gap.pth.tar',
}

UNIT_MAP = {
    'formation_energy': 'eV/atom',
    'band_gap':         'eV',
}


class Normalizer:
    """还原归一化"""
    def __init__(self, state_dict):
        self.mean = torch.tensor(state_dict['mean'])
        self.std  = torch.tensor(state_dict['std'])

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean


class CGCNNPredictor(BasePredictor):
    name = 'CGCNN'

    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.models     = {}
        self.normalizers = {}

        for prop, fname in MODEL_FILES.items():
            model_path = PRETRAINED_DIR / fname
            if not os.path.exists(model_path):
                print(f'CGCNN {prop} 权重文件不存在: {model_path}')
                continue
            try:
                checkpoint = torch.load(
                    model_path,
                    map_location=self.device,
                    weights_only=False,
                )
                args = checkpoint['args']

                # orig_atom_fea_len 从 state_dict 推断
                state_dict = checkpoint['state_dict']
                orig_atom_fea_len = state_dict['embedding.weight'].shape[1]
                nbr_fea_len       = state_dict['convs.0.fc_full.weight'].shape[1] - 2 * args['atom_fea_len']

                model = CrystalGraphConvNet(
                    orig_atom_fea_len = orig_atom_fea_len,
                    nbr_fea_len       = nbr_fea_len,
                    atom_fea_len      = args.get('atom_fea_len', 64),
                    n_conv            = args.get('n_conv', 4),
                    h_fea_len         = args.get('h_fea_len', 32),
                    n_h               = args.get('n_h', 1),
                )
                model.load_state_dict(state_dict)
                model.to(self.device)
                model.eval()
                self.models[prop] = model

                # 加载归一化参数
                if 'normalizer' in checkpoint:
                    self.normalizers[prop] = Normalizer(checkpoint['normalizer'])

                print(f'CGCNN {prop} 加载成功')
            except Exception as e:
                print(f'CGCNN {prop} 加载失败: {e}')

    def predict(self, structure: Structure, property_name: str) -> dict:
        if property_name not in self.models:
            raise ValueError(f'CGCNN 不支持预测 {property_name}')

        import tempfile
        from pymatgen.io.cif import CifWriter

        model      = self.models[property_name]
        normalizer = self.normalizers.get(property_name)

        with tempfile.TemporaryDirectory() as tmpdir:
            cif_path = os.path.join(tmpdir, 'structure.cif')
            CifWriter(structure).write_file(cif_path)

            # CGCNN 需要 id_prop.csv
            with open(os.path.join(tmpdir, "id_prop.csv"), "w") as f:
                f.write("structure,0\n")

            import shutil
            shutil.copy(
                CGCNN_DIR / 'cgcnn' / 'atom_init.json',
                os.path.join(tmpdir, 'atom_init.json')
            )
            dataset = CIFData(tmpdir)
            loader  = torch.utils.data.DataLoader(
                dataset,
                batch_size=1,
                collate_fn=collate_pool,
            )

            with torch.no_grad():
                for input_data, _, _ in loader:
                    atom_fea        = input_data[0].to(self.device)
                    nbr_fea         = input_data[1].to(self.device)
                    nbr_fea_idx     = input_data[2].to(self.device)
                    crystal_atom_idx = input_data[3]

                    output = model(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)

                    if normalizer:
                        value = normalizer.denorm(output).item()
                    else:
                        value = output.item()

        return {
            'value': round(float(value), 4),
            'unit':  UNIT_MAP[property_name],
        }

    def is_available(self) -> bool:
        return len(self.models) > 0
