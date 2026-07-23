"""
多态性检测节点：判断化学式是否存在多种已知晶体结构
策略：先查内置字典，未命中时用LLM判断
"""
from utils.polymorph_dict import POLYMORPH_DICT
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import json

llm = ChatOpenAI(model="gpt-4o", temperature=0)

LLM_POLYMORPH_PROMPT = ChatPromptTemplate.from_template("""
化学式 {formula} 是否存在多种已知晶体结构（多态性）？
如果存在，列出主要的相。返回JSON格式：
{{"is_polymorph": true, "phases": ["立方相", "四方相", "正交相"], "note": "由LLM推断，可能不完整"}}
如果不存在：
{{"is_polymorph": false, "phases": [], "note": ""}}
只返回JSON。
""")


def check_polymorph(state: dict) -> dict:
    formula = state.get("formula")
    if not formula:
        return {"error": "未能识别化学式"}

    # 第一步：查内置字典
    if formula in POLYMORPH_DICT:
        phases = POLYMORPH_DICT[formula]
        if len(phases) > 1:
            return {"polymorphs": phases}
        return {"polymorphs": None}  # 唯一结构，直接进入下一步

    # 第二步：LLM判断（冷门材料兜底）
    chain = LLM_POLYMORPH_PROMPT | llm
    result = chain.invoke({"formula": formula})

    try:
        parsed = json.loads(result.content)
        if parsed.get("is_polymorph"):
            phases = parsed.get("phases", [])
            # 附加LLM推断标注
            phases_with_note = [f"{p}（LLM推断）" for p in phases]
            return {"polymorphs": phases_with_note}
        return {"polymorphs": None}
    except Exception:
        return {"polymorphs": None}  # 解析失败则假设无多态，继续流程
