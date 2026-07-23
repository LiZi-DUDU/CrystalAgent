"""
第二阶段：学习路由
轻量分类器，输入晶系+性质+元素组成，输出最优模型
需要先运行 scripts/train_learned_router.py 训练后才能启用
"""
from pathlib import Path
import pickle


class LearnedRouter:
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.classifier = self._load_classifier()

    def _load_classifier(self):
        if not self.model_path.exists():
            return None
        with open(self.model_path, "rb") as f:
            return pickle.load(f)

    def is_available(self) -> bool:
        return self.classifier is not None

    def route(self, crystal_system: str, prop: str, formula: str) -> str:
        """
        输入特征：晶系编码 + 性质编码 + 元素组成
        输出：最优模型名称
        """
        features = self._featurize(crystal_system, prop, formula)
        model_idx = self.classifier.predict([features])[0]
        return self.classifier.classes_[model_idx]

    def _featurize(self, crystal_system: str, prop: str, formula: str) -> list:
        """
        特征向量构建，后续可替换为更丰富的表示
        """
        CRYSTAL_SYSTEMS = ["三斜", "单斜", "正交", "四方", "三方", "六方", "立方"]
        PROPERTIES      = ["band_gap", "formation_energy", "elastic_modulus", "phonon"]

        cs_vec   = [1 if crystal_system == cs else 0 for cs in CRYSTAL_SYSTEMS]
        prop_vec = [1 if prop == p else 0 for p in PROPERTIES]

        # 简单元素组成特征（后续可替换为元素嵌入）
        from pymatgen.core import Composition
        comp      = Composition(formula)
        elem_feat = [comp.get(el, 0) for el in ["O", "N", "S", "F", "Fe", "Cu", "Ti", "Ba"]]

        return cs_vec + prop_vec + elem_feat
