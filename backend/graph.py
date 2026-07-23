"""
主流程图：定义智能体的节点和边
"""
from typing import TypedDict, Optional, List
from langgraph.graph import StateGraph, END

from nodes.parser import parse_input
from nodes.polymorph import check_polymorph
from nodes.structure import parse_structure
from nodes.router import route_models
from nodes.predictor import run_prediction
from nodes.reporter import generate_report


# ── 全局状态定义 ──────────────────────────────────────────
class AgentState(TypedDict):
    # 输入
    user_input: str                      # 用户原始输入
    structure_file: Optional[bytes]      # 上传的结构文件

    # 解析结果
    formula: Optional[str]               # 化学式
    target_properties: List[str]         # 目标性质列表，如 ["band_gap", "formation_energy"]

    # 多态性检测
    polymorphs: Optional[List[str]]      # 候选相列表
    awaiting_user_choice: bool           # 是否等待用户选择相

    # 结构信息
    crystal_system: Optional[str]        # 晶系（spglib确定）
    space_group: Optional[str]           # 空间群

    # 路由结果
    # 格式: {"band_gap": "ALIGNN", "formation_energy": "MEGNet"}
    model_assignments: Optional[dict]
    router_mode: Optional[str]           # "table" | "learned" | "ensemble"

    # 预测结果
    # 格式: {"band_gap": {"value": 2.34, "unit": "eV", "model": "ALIGNN", "mae": 0.31}}
    predictions: Optional[dict]

    # 最终报告
    report: Optional[str]

    # 错误信息
    error: Optional[str]


# ── 条件分支函数 ──────────────────────────────────────────
def should_ask_polymorph(state: AgentState) -> str:
    """多态性检测后的分支：需要用户选择 or 直接上传文件"""
    if state.get("polymorphs") and not state.get("structure_file"):
        return "ask_user"
    return "parse_structure"


def has_structure_file(state: AgentState) -> str:
    """是否已有结构文件"""
    if state.get("structure_file"):
        return "parse_structure"
    return "wait_for_file"


# ── 构建 LangGraph 流程图 ─────────────────────────────────
def build_graph():
    graph = StateGraph(AgentState)

    # 注册节点
    graph.add_node("parse_input",      parse_input)
    graph.add_node("check_polymorph",  check_polymorph)
    graph.add_node("parse_structure",  parse_structure)
    graph.add_node("route_models",     route_models)
    graph.add_node("run_prediction",   run_prediction)
    graph.add_node("generate_report",  generate_report)

    # 定义边
    graph.set_entry_point("parse_input")
    graph.add_edge("parse_input", "check_polymorph")

    # 多态性检测后的条件分支
    graph.add_conditional_edges(
        "check_polymorph",
        should_ask_polymorph,
        {
            "ask_user":       END,           # 暂停，等待用户选择相后重新触发
            "parse_structure": "parse_structure",
        }
    )

    graph.add_edge("parse_structure", "route_models")
    graph.add_edge("route_models",    "run_prediction")
    graph.add_edge("run_prediction",  "generate_report")
    graph.add_edge("generate_report", END)

    return graph.compile()


agent = build_graph()
