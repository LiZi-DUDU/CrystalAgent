"""
FastAPI 入口：提供REST API接口
"""
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn

from graph import agent

app = FastAPI(title="晶体性质预测智能体")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite开发服务器
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictResponse(BaseModel):
    report:          Optional[str]
    predictions:     Optional[dict]
    crystal_system:  Optional[str]
    space_group:     Optional[str]
    router_mode:     Optional[str]
    polymorphs:      Optional[list]   # 若存在多态，返回候选相列表
    error:           Optional[str]


@app.post("/predict", response_model=PredictResponse)
async def predict(
    user_input:     str        = Form(...),
    structure_file: UploadFile = File(None),
    selected_phase: str        = Form(None),   # 用户选择的相
):
    state = {"user_input": user_input}

    if structure_file:
        state["structure_file"] = await structure_file.read()

    if selected_phase:
        state["selected_phase"] = selected_phase

    result = await agent.ainvoke(state)

    # 若检测到多态且等待用户选择，返回候选相列表
    if result.get("awaiting_user_choice"):
        return PredictResponse(polymorphs=result.get("polymorphs"))

    return PredictResponse(
        report         = result.get("report"),
        predictions    = result.get("predictions"),
        crystal_system = result.get("crystal_system"),
        space_group    = result.get("space_group"),
        router_mode    = result.get("router_mode"),
        error          = result.get("error"),
    )


@app.get("/routing_table")
def get_routing_table():
    """查看当前路由表（调试用）"""
    import json
    from pathlib import Path
    path = Path("data/routing_table/mae_matrix.json")
    if path.exists():
        return json.loads(path.read_text())
    return {}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
