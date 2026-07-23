"""
报告生成节点：LLM根据预测结果生成三层结构化报告
"""
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import json

llm = ChatOpenAI(model="gpt-4o", temperature=0.2)

REPORT_PROMPT = ChatPromptTemplate.from_template("""
根据以下晶体性质预测结果，生成一份结构化报告。

材料信息：
- 化学式：{formula}
- 晶系：{crystal_system}（{space_group}）
- 路由模式：{router_mode}

预测结果：
{predictions_json}

请按以下三层格式生成报告：

【第一层：预测结果】
列出每个性质的预测值和单位，简洁清晰。

【第二层：可信度信息】
对每个性质说明：调用模型、该晶系历史MAE。
若为集成模式，说明各模型权重及触发原因（路由表样本不足）。

【第三层：局限性说明】
- 说明当前晶系的训练集覆盖情况
- 若结构存在特殊性，给出警告
- 若为集成模式，注明不确定度较高

语言简洁专业，面向材料科学研究者。
""")


def generate_report(state: dict) -> dict:
    predictions = state.get("predictions", {})

    # 整理预测结果为可读格式
    predictions_summary = []
    for prop, result in predictions.items():
        if "error" in result:
            predictions_summary.append(f"{prop}: 预测失败 - {result['error']}")
        else:
            predictions_summary.append(
                f"{prop}: {result['value']} {result['unit']} "
                f"（模型: {result['model']}，MAE: {result.get('mae', 'N/A')}）"
            )

    chain  = REPORT_PROMPT | llm
    result = chain.invoke({
        "formula":          state.get("formula", "未知"),
        "crystal_system":   state.get("crystal_system", "未知"),
        "space_group":      state.get("space_group", "未知"),
        "router_mode":      state.get("router_mode", "table"),
        "predictions_json": "\n".join(predictions_summary),
    })

    return {"report": result.content}
