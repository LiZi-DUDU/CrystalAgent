"""
路由节点：三阶段路由逻辑
第一阶段：查表路由（主路径）
第二阶段：学习路由（升级替代，需训练分类器后启用）
第三阶段：集成模式（兜底）
"""
from router.table_router import TableRouter
from router.learned_router import LearnedRouter
from router.ensemble_router import EnsembleRouter

table_router    = TableRouter("data/routing_table/mae_matrix.json")
learned_router  = LearnedRouter("data/routing_table/learned_router.pkl")
ensemble_router = EnsembleRouter()

# 样本数阈值：低于此值触发集成模式
MIN_SAMPLE_THRESHOLD = 50


def route_models(state: dict) -> dict:
    crystal_system     = state.get("crystal_system")
    target_properties  = state.get("target_properties", [])

    model_assignments = {}
    router_mode = "table"

    for prop in target_properties:
        # 第一阶段：查表路由
        result = table_router.route(crystal_system, prop)

        if result["sample_count"] >= MIN_SAMPLE_THRESHOLD:
            # 尝试第二阶段：学习路由（如果分类器已训练）
            if learned_router.is_available():
                formula  = state.get("formula", "")
                model    = learned_router.route(crystal_system, prop, formula)
                router_mode = "learned"
            else:
                model       = result["best_model"]
                router_mode = "table"
        else:
            # 第三阶段：集成模式兜底
            model       = "ensemble"
            router_mode = "ensemble"

        model_assignments[prop] = {
            "model":        model,
            "mae":          result.get("mae"),
            "sample_count": result.get("sample_count"),
        }

    return {
        "model_assignments": model_assignments,
        "router_mode":       router_mode,
    }
