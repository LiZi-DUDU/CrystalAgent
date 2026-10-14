#!/usr/bin/env python3
import hashlib
import json
import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path
import os

base = Path(os.environ.get('CRYSTALAGENT_ROOT', Path(__file__).resolve().parents[1])).resolve()
run_dir = base / 'data/benchmark/crystalagent_e2e_kg_v2_1_seed42'

with (base / 'data/knowledge_graph/sci_tool_kg.pkl').open('rb') as handle:
    graph = pickle.load(handle)

def norm_prop(value):
    if value in ('formation_energy', 'formation', 'energy'):
        return 'formation_energy'
    if value in ('band_gap', 'band', 'gap'):
        return 'band_gap'
    return value

route_maps = defaultdict(dict)
route_meta = defaultdict(dict)
for source, target, data in graph.edges(data=True):
    if data.get('relation') != 'BEST_FOR' or not source.startswith('CrystalSystem:'):
        continue
    prop = norm_prop(data.get('property'))
    crystal_system = source.replace('CrystalSystem:', '').split('_')[0]
    model = target.replace('Model:', '')
    route_maps[prop][crystal_system] = model
    route_meta[prop][crystal_system] = {
        'model': model,
        'mae': data.get('mae'),
        'sample_count': data.get('sample_count'),
        'split': data.get('split'),
        'frozen': data.get('frozen'),
        'source': data.get('source'),
    }

report = {'kg_routes': dict(route_meta)}
for prop in ('band_gap', 'formation_energy'):
    rows_path = run_dir / f'crystalagent_e2e_{prop}_seed42.jsonl'
    summary_path = run_dir / f'crystalagent_e2e_{prop}_seed42_summary.json'
    manifest_path = run_dir / f'manifest_{prop}_seed42.json'
    rows = [json.loads(line) for line in rows_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    rows.sort(key=lambda row: row['i'])
    ids = [row['material_id'] for row in rows]
    digest = hashlib.sha256('\n'.join(ids).encode('utf-8')).hexdigest()
    route_mismatches = [
        {
            'material_id': row['material_id'],
            'crystal_system': row.get('crystal_system'),
            'recorded_model': row.get('model'),
            'kg_model': route_maps[prop].get(row.get('crystal_system')),
        }
        for row in rows
        if row.get('model') != route_maps[prop].get(row.get('crystal_system'))
    ]
    pairs = [
        (float(row['pred']), float(row['gt'])) for row in rows
        if row.get('pred') is not None and math.isfinite(float(row['pred']))
    ]
    clipped = sum(
        any('截断为 0' in warning for warning in
            row.get('stages', {}).get('model_execute', {}).get('warnings', []))
        for row in rows
    )
    report[prop] = {
        'n': len(rows),
        'unique_id_n': len(set(ids)),
        'manifest_n': manifest['n'],
        'row_id_sha256': digest,
        'manifest_id_sha256': manifest['ordered_material_id_sha256'],
        'hash_matches': digest == manifest['ordered_material_id_sha256'],
        'prediction_success_n': sum(bool(row.get('prediction_success')) for row in rows),
        'end_to_end_success_n': sum(bool(row.get('end_to_end_success')) for row in rows),
        'mae_recomputed': sum(abs(pred - gt) for pred, gt in pairs) / len(pairs),
        'rmse_recomputed': math.sqrt(sum((pred - gt) ** 2 for pred, gt in pairs) / len(pairs)),
        'summary_mae': summary['mae'],
        'summary_rmse': summary['rmse'],
        'model_counts': dict(Counter(row.get('model') for row in rows)),
        'route_mismatch_n': len(route_mismatches),
        'route_mismatches_first_10': route_mismatches[:10],
        'negative_clipped_n': clipped,
        'failure_error_count': sum(len(row.get('errors', [])) for row in rows),
    }

old_path = base / 'data/benchmark/llm4mat_formation_energy_both_42.jsonl'
old_rows = [json.loads(line) for line in old_path.read_text(encoding='utf-8').splitlines() if line.strip()]
new_rows = [json.loads(line) for line in (run_dir / 'crystalagent_e2e_formation_energy_seed42.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
old_by_id = {row['material_id']: row for row in old_rows}
new_by_id = {row['material_id']: row for row in new_rows}
changes = []
for material_id in sorted(set(old_by_id) & set(new_by_id)):
    old_model = (old_by_id[material_id].get('agent') or {}).get('model')
    new_model = new_by_id[material_id].get('model')
    if old_model != new_model:
        changes.append((material_id, old_model, new_model))
report['formation_old_new'] = {
    'same_id_set': set(old_by_id) == set(new_by_id),
    'old_model_counts': dict(Counter((row.get('agent') or {}).get('model') for row in old_rows)),
    'new_model_counts': dict(Counter(row.get('model') for row in new_rows)),
    'route_change_n': len(changes),
}

print(json.dumps(report, ensure_ascii=False, indent=2))
