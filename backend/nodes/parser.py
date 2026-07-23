"""
输入解析节点：用LLM从自然语言/化学式中提取化学式和目标性质
"""
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import json

llm = ChatOpenAI(model="gpt-4o", temperature=0)

PARSE_PROMPT = ChatPromptTemplate.from_template("""
从用户输入中提取以下信息，返回JSON格式：
- formula: 化学式（如 BaTiO3）
- target_properties: 目标性质列表，从以下选项中选择：
  band_gap（带隙）、formation_energy（形成能）、
  elastic_modulus（弹性模量）、phonon（声子性质）

用户输入：{user_input}

只返回JSON，不要其他内容。示例：
{{"formula": "BaTiO3", "target_properties": ["band_gap", "formation_energy"]}}
""")


def parse_input(state: dict) -> dict:
    chain = PARSE_PROMPT | llm
    result = chain.invoke({"user_input": state["user_input"]})

    try:
        parsed = json.loads(result.content)
        return {
            "formula": parsed.get("formula"),
            "target_properties": parsed.get("target_properties", []),
        }
    except Exception:
        return {"error": f"无法解析输入：{state['user_input']}"}
