import asyncio
from fastapi.responses import HTMLResponse
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import uvicorn

try:
    from .graph_orchestrator import GraphOrchestrator
except ImportError:  # Support ``python backend/main.py``.
    from graph_orchestrator import GraphOrchestrator

app = FastAPI(title='Crystal Agent - Knowledge Graph Orchestrator')
@app.get('/', response_class=HTMLResponse)
def index():
    import os
    path = os.path.join(os.path.dirname(__file__), 'static', 'index.html')
    return open(path, encoding='utf-8').read()


app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:5173'],
    allow_methods=['*'],
    allow_headers=['*'],
)

# Initialize orchestrator (lazy-loads KG and models)
orchestrator = GraphOrchestrator()


class PredictResponse(BaseModel):
    report:          Optional[str] = None
    predictions:     Optional[dict] = None
    formula:         Optional[str] = None
    crystal_system:  Optional[str] = None
    space_group:     Optional[str] = None
    plan:            Optional[list] = None
    error:           Optional[str] = None


@app.post('/predict', response_model=PredictResponse)
async def predict(
    user_input:     str        = Form(...),
    structure_file: UploadFile = File(None),
    formula:        str        = Form(None),
    selected_mp_id: str        = Form(None),
):
    structure_bytes = await structure_file.read() if structure_file else None

    # Run sync orchestrator in thread pool to avoid blocking event loop
    result = await asyncio.to_thread(
        orchestrator.run,
        user_input=user_input,
        structure_file=structure_bytes,
        selected_mp_id=selected_mp_id,
    )

    if result.get('error'):
        return PredictResponse(error=str(result['error']))

    return PredictResponse(
        report         = result.get('report'),
        predictions    = result.get('results'),
        formula        = result.get('formula'),
        crystal_system = result.get('crystal_system'),
        space_group    = result.get('space_group'),
        plan           = result.get('plan'),
    )


@app.post("/search_cif")
async def search_cif(formula: str = Form(...)):
    candidates = orchestrator.find_cif_by_formula(formula)
    return {"formula": formula, "count": len(candidates), "candidates": candidates}


@app.get('/routing_table')
def get_routing_table():
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    path = root / 'data' / 'routing_table' / 'mae_matrix.json'
    if path.exists():
        return json.loads(path.read_text())
    return {}


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8004)
