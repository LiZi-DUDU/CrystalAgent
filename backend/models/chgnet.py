"""
CHGNet 模型接口
CHGNet 是势能面模型，设计目标是总能量/力/应力预测
不适合直接预测形成能、带隙等性质
当前版本暂不接入性质预测路由
"""
from models.base import BasePredictor
from pymatgen.core import Structure


class CHGNetPredictor(BasePredictor):
    name = "CHGNet"
    supported_properties = []  # 暂不支持任何性质预测

    def __init__(self):
        from chgnet.model import CHGNet
        self.model = CHGNet.load()

    def predict(self, structure: Structure, property_name: str) -> dict:
        raise ValueError(
            f"CHGNet 不支持直接预测 {property_name}。"
            f"CHGNet 是势能面模型，适合分子动力学模拟，不适合性质预测任务。"
        )

    def is_available(self) -> bool:
        return False  # 不参与路由