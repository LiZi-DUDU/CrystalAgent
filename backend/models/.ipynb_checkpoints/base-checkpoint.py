"""
GNN模型基类：所有模型封装继承此类
"""
from abc import ABC, abstractmethod
from pymatgen.core import Structure


class BasePredictor(ABC):
    name: str = "base"

    @abstractmethod
    def predict(self, structure: Structure, property_name: str) -> dict:
        """
        返回格式:
        {
            "value": 2.34,
            "unit": "eV",
            "confidence": 0.85   # 可选，模型输出的置信度
        }
        """
        pass

    def is_available(self) -> bool:
        """检查模型是否已加载，子类可覆盖"""
        return True
