#!/usr/bin/env python3
"""Run one deterministic CIF->KG->GNN smoke case for every frozen-route model."""

import json
import os
import sys
import time
from pathlib import Path

BASE = Path(os.environ.get('CRYSTALAGENT_ROOT', Path(__file__).resolve().parents[1])).resolve()
sys.path.insert(0, str(BASE / 'backend'))
os.environ.setdefault('DEEPSEEK_API_KEY', 'smoke-test-placeholder')

from graph_orchestrator import GraphOrchestrator


NEEDED = {
    ('band_gap', 'CHGNet'),
    ('band_gap', 'ALIGNN'),
    ('formation_energy', 'SevenNet'),
    ('formation_energy', 'MEGNet'),
    ('formation_energy', 'M3GNet'),
}

CURRENT_ROUTES = {
    'band_gap': {
        '三斜': 'CHGNet', '单斜': 'CHGNet', '正交': 'CHGNet',
        '四方': 'ALIGNN', '三方': 'ALIGNN', '六方': 'ALIGNN', '立方': 'ALIGNN',
    },
    'formation_energy': {
        '三斜': 'SevenNet', '单斜': 'MEGNet', '正交': 'SevenNet',
        '四方': 'M3GNet', '三方': 'MEGNet', '六方': 'MEGNet', '立方': 'M3GNet',
    },
}


def main():
    orchestrator = GraphOrchestrator()
    selected = {}
    for prop in ('band_gap', 'formation_energy'):
        manifest = json.load(open(
            BASE / 'data' / 'benchmark' / 'crystalagent_e2e_kg_v2_1_seed42'
            / f'manifest_{prop}_seed42.json'
        ))
        task_by_id = {task['material_id']: task for task in manifest['tasks']}
        old_rows = [
            json.loads(line) for line in open(
                BASE / 'data' / 'benchmark' / f'llm4mat_{prop}_both_42.jsonl'
            )
        ]
        candidates = []
        seen_models = set()
        for row in old_rows:
            crystal_system = row.get('agent', {}).get('crystal_system')
            model = CURRENT_ROUTES[prop].get(crystal_system)
            if (prop, model) in NEEDED and model not in seen_models:
                candidates.append(task_by_id[row['material_id']])
                seen_models.add(model)
        for task in candidates:
            with open(task['cif_path'], 'rb') as handle:
                structure, crystal_system, space_group = orchestrator.parse_structure(handle.read())
            plan = orchestrator.kg_plan(crystal_system, [prop])
            model = plan[0]['model']
            if (prop, model) in NEEDED and (prop, model) not in selected:
                selected[(prop, model)] = (task, structure, crystal_system, space_group, plan)

    missing = NEEDED - set(selected)
    if missing:
        raise RuntimeError(f'No smoke task found for {sorted(missing)}')

    results = []
    for prop, model in sorted(NEEDED):
        task, structure, crystal_system, space_group, plan = selected[(prop, model)]
        started = time.time()
        output = orchestrator.execute(structure, crystal_system, plan)[prop]
        record = {
            'property': prop,
            'model': model,
            'material_id': task['material_id'],
            'crystal_system': crystal_system,
            'space_group': space_group,
            'pred': output.get('value'),
            'warnings': output.get('warnings', []),
            'seconds': time.time() - started,
            'ok': output.get('value') is not None,
        }
        results.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    if not all(record['ok'] for record in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
