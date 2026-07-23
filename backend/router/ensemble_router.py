"""
第三阶段：集成模式（兜底）
当路由表样本不足时触发，调用所有模型加权融合
"""

ALL_MODELS = ["CGCNN", "MEGNet", "ALIGNN", "M3GNet", "CHGNet"]


class EnsembleRouter:
    def get_models(self) -> list:
        return ALL_MODELS

    def compute_weights(self, all_models_mae: dict) -> dict:
        """
        按历史MAE的倒数分配权重，MAE越小权重越大
        如果没有MAE数据，等权分配
        """
        if not all_models_mae:
            n = len(ALL_MODELS)
            return {m: 1.0 / n for m in ALL_MODELS}

        inv_mae = {}
        for model in ALL_MODELS:
            mae = all_models_mae.get(model)
            inv_mae[model] = (1.0 / mae) if mae and mae > 0 else 0.1

        total = sum(inv_mae.values())
        return {m: v / total for m, v in inv_mae.items()}
