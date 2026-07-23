"""
预测节点：根据路由结果调用对应模型进行预测
集成模式下调用所有模型并加权融合
"""
from pymatgen.core import Structure
from pymatgen.io.cif import CifParser
from router.ensemble_router import EnsembleRouter

# 模型注册表（按需加载）
def _load_models():
    from models.cgcnn   import CGCNNPredictor
    from models.megnet  import MEGNetPredictor
    from models.alignn  import ALIGNNPredictor
    from models.m3gnet  import M3GNetPredictor
    from models.chgnet  import CHGNetPredictor
    return {
        "CGCNN":  CGCNNPredictor(),
        "MEGNet": MEGNetPredictor(),
        "ALIGNN": ALIGNNPredictor(),
        "M3GNet": M3GNetPredictor(),
        "CHGNet": CHGNetPredictor(),
    }

ensemble_router = EnsembleRouter()
_models = None


def _get_models():
    global _models
    if _models is None:
        _models = _load_models()
    return _models


def run_prediction(state: dict) -> dict:
    file_bytes        = state.get("structure_file")
    model_assignments = state.get("model_assignments", {})

    # 解析结构
    structure = _parse_structure(file_bytes)
    if structure is None:
        return {"error": "结构文件解析失败，无法进行预测"}

    models      = _get_models()
    predictions = {}

    for prop, assignment in model_assignments.items():
        model_name = assignment["model"]

        if model_name == "ensemble":
            # 集成模式：所有模型都跑，加权融合
            result = _run_ensemble(structure, prop, assignment.get("all_models_mae", {}))
        else:
            # 单模型预测
            predictor = models.get(model_name)
            if predictor is None:
                predictions[prop] = {"error": f"模型 {model_name} 不可用"}
                continue
            raw = predictor.predict(structure, prop)
            result = {
                "value": raw["value"],
                "unit":  raw["unit"],
                "model": model_name,
                "mae":   assignment.get("mae"),
                "mode":  "single",
            }

        predictions[prop] = result

    return {"predictions": predictions}


def _run_ensemble(structure, prop: str, all_models_mae: dict) -> dict:
    models   = _get_models()
    weights  = ensemble_router.compute_weights(all_models_mae)
    results  = {}

    for model_name, predictor in models.items():
        try:
            raw = predictor.predict(structure, prop)
            results[model_name] = raw["value"]
        except Exception:
            pass

    if not results:
        return {"error": "所有模型预测失败"}

    # 加权融合
    total_weight   = sum(weights[m] for m in results)
    fused_value    = sum(results[m] * weights[m] for m in results) / total_weight
    unit           = _get_unit(prop)

    return {
        "value":   round(fused_value, 4),
        "unit":    unit,
        "model":   "ensemble",
        "weights": {m: round(weights[m], 3) for m in results},
        "mode":    "ensemble",
    }


def _parse_structure(file_bytes: bytes):
    try:
        content = file_bytes.decode("utf-8", errors="ignore")
        parser  = CifParser.from_str(content)
        return parser.parse_structures()[0]
    except Exception:
        return None


def _get_unit(prop: str) -> str:
    units = {
        "band_gap":          "eV",
        "formation_energy":  "eV/atom",
        "elastic_modulus":   "GPa",
        "phonon":            "THz",
    }
    return units.get(prop, "")
