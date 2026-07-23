# 晶体性质预测智能体

基于 LangGraph + FastAPI + React 的多模型晶体性质预测系统。

## 项目结构

```
crystal_agent/
├── backend/
│   ├── main.py                  # FastAPI 入口
│   ├── graph.py                 # LangGraph 主流程图
│   ├── nodes/                   # 各流程节点
│   │   ├── parser.py            # 输入解析节点（LLM提取化学式+性质）
│   │   ├── polymorph.py         # 多态性检测节点
│   │   ├── structure.py         # 结构解析节点（spglib确定晶系）
│   │   ├── router.py            # 路由节点（三阶段路由）
│   │   ├── predictor.py         # 模型预测节点
│   │   └── reporter.py          # 报告生成节点（LLM）
│   ├── models/                  # 各GNN模型封装
│   │   ├── base.py              # 模型基类
│   │   ├── cgcnn.py
│   │   ├── megnet.py
│   │   ├── alignn.py
│   │   ├── m3gnet.py
│   │   ├── chgnet.py
│   │   └──補全模型.py           # 空白晶系补全模型
│   ├── router/
│   │   ├── table_router.py      # 第一阶段：查表路由
│   │   ├── learned_router.py    # 第二阶段：学习路由
│   │   └── ensemble_router.py   # 第三阶段：集成兜底
│   └── utils/
│       ├── spglib_utils.py      # 晶系识别工具
│       └── polymorph_dict.py    # 多态性字典
├── frontend/
│   └── src/
│       ├── components/
│       │   ├── InputPanel.jsx   # 用户输入面板
│       │   ├── UploadBox.jsx    # 结构文件上传
│       │   └── ReportView.jsx   # 预测报告展示
│       └── pages/
│           └── Home.jsx
├── config/
│   └── settings.py              # 全局配置（模型路径、API key等）
├── data/
│   ├── routing_table/
│   │   └── mae_matrix.json      # 晶系×性质×模型 MAE矩阵
│   └── benchmark/               # 基准实验数据（按晶系分层）
├── scripts/
│   ├── build_routing_table.py   # 跑基准实验、填路由表
│   └── evaluate_system.py       # 系统评估脚本
└── requirements.txt
```

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 启动后端
cd backend && uvicorn main:app --reload

# 启动前端
cd frontend && npm install && npm run dev
```
