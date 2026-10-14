#!/usr/bin/env python3
"""Instrumented end-to-end CrystalAgent benchmark on frozen seed-42 subsets.

The property-specific material IDs are inherited from the original LLM4Mat-Bench
comparison files, then revalidated against ``LLM4Mat-Bench/test.csv``.  Each
record traverses natural-language parsing, CIF parsing, frozen KG routing, model
execution, safety checks and report generation.
"""

from __future__ import annotations

import argparse
import ctypes
import csv
import gc
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path


BASE = Path(os.environ.get('CRYSTALAGENT_ROOT', Path(__file__).resolve().parents[1])).resolve()
TEST_CSV = BASE / 'LLM4Mat-Bench' / 'test.csv'
STRUCTURES = BASE / 'data' / 'structures'
BENCHMARK_DIR = BASE / 'data' / 'benchmark'
RUN_DIR = BENCHMARK_DIR / 'crystalagent_e2e_kg_v2_1_seed42'
BACKEND_DIR = BASE / 'backend'
sys.path.insert(0, str(BACKEND_DIR))

PROPS = ('band_gap', 'formation_energy')
GT_COLUMN = {
    'band_gap': 'band_gap',
    'formation_energy': 'formation_energy_per_atom',
}
DISPLAY = {
    'band_gap': 'band gap',
    'formation_energy': 'formation energy',
}
UNIT = {
    'band_gap': 'eV',
    'formation_energy': 'eV/atom',
}
VALID_RANGE = {
    'band_gap': (0.0, 20.0),
    'formation_energy': (-10.0, 5.0),
}
FORMATION_ROUTE_BY_CRYSTAL_SYSTEM = {
    '三斜': 'SevenNet',
    '单斜': 'MEGNet',
    '正交': 'ALIGNN',
    '四方': 'M3GNet',
    '三方': 'MEGNet',
    '六方': 'MEGNet',
    '立方': 'M3GNet',
}


def load_private_environment() -> None:
    """Load a local protected key file without printing or persisting its value."""
    if os.environ.get('DEEPSEEK_API_KEY'):
        return
    for path in (
        BASE / '.env',
        Path('/root/.config/crystalagent/deepseek.env'),
        BASE / 'README.md',  # user-provided legacy location; value is never printed
    ):
        if not path.exists():
            continue
        for raw_line in path.read_text(encoding='utf-8').splitlines():
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('export '):
                line = line[7:].strip()
            if '=' not in line:
                continue
            key, value = line.split('=', 1)
            if key.strip() == 'DEEPSEEK_API_KEY':
                # README lines may contain explanatory prose after the key.
                token = value.strip().split()[0].strip('"').strip("'")
                os.environ['DEEPSEEK_API_KEY'] = token
                return


def json_dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
    tmp.replace(path)


def sha256_ids(rows: list[dict]) -> str:
    payload = '\n'.join(row['material_id'] for row in rows).encode()
    return hashlib.sha256(payload).hexdigest()


def original_seed42_path(prop: str) -> Path:
    return BENCHMARK_DIR / f'llm4mat_{prop}_both_42.jsonl'


def manifest_path(prop: str) -> Path:
    return RUN_DIR / f'manifest_{prop}_seed42.json'


def prepare_manifest(prop: str) -> dict:
    source_path = original_seed42_path(prop)
    source_rows = [json.loads(line) for line in source_path.read_text(encoding='utf-8').splitlines() if line]
    ordered_ids = [row['material_id'] for row in source_rows]
    if len(ordered_ids) != 1000 or len(set(ordered_ids)) != 1000:
        raise ValueError(f'{source_path} does not contain 1,000 unique material IDs')

    wanted = set(ordered_ids)
    indexed = {}
    csv.field_size_limit(2 ** 31 - 1)
    with TEST_CSV.open(encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            material_id = row['material_id']
            if material_id not in wanted or material_id in indexed:
                continue
            raw_gt = row[GT_COLUMN[prop]].strip()
            if not raw_gt:
                raise ValueError(f'Missing {prop} label for {material_id}')
            indexed[material_id] = {
                'material_id': material_id,
                'formula': row['formula_pretty'],
                'gt': float(raw_gt),
                'cif_path': str(STRUCTURES / f'{material_id}.cif'),
            }
            if len(indexed) == len(wanted):
                break

    missing = [material_id for material_id in ordered_ids if material_id not in indexed]
    if missing:
        raise ValueError(f'{len(missing)} selected IDs were not found in test.csv: {missing[:5]}')

    tasks = [indexed[material_id] for material_id in ordered_ids]
    for task in tasks:
        if not Path(task['cif_path']).exists():
            raise FileNotFoundError(task['cif_path'])

    old_gt = {row['material_id']: float(row['gt']) for row in source_rows}
    mismatches = [
        task['material_id'] for task in tasks
        if not math.isclose(task['gt'], old_gt[task['material_id']], rel_tol=0.0, abs_tol=1e-12)
    ]
    if mismatches:
        raise ValueError(f'Ground-truth drift for {len(mismatches)} IDs: {mismatches[:5]}')

    manifest = {
        'property': prop,
        'subset_seed': 42,
        'selection_protocol': 'property-specific 1,000-structure subset inherited from the original comparison',
        'source_csv': str(TEST_CSV),
        'source_id_file': str(source_path),
        'ordered_material_id_sha256': sha256_ids(tasks),
        'n': len(tasks),
        'tasks': tasks,
    }
    json_dump(manifest_path(prop), manifest)
    return manifest


def load_manifest(prop: str) -> dict:
    path = manifest_path(prop)
    if not path.exists():
        return prepare_manifest(prop)
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if manifest['n'] != 1000 or sha256_ids(manifest['tasks']) != manifest['ordered_material_id_sha256']:
        raise ValueError(f'Invalid manifest: {path}')
    return manifest


def retry_call(function, attempts: int, stage: str):
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            return function(), attempt, None
        except Exception as exc:  # external API and model packages expose heterogeneous errors
            last_error = f'{type(exc).__name__}: {str(exc)}'
            if attempt < attempts:
                time.sleep(min(8, 2 ** attempt))
    return None, attempts, {'stage': stage, 'message': last_error[:1000]}


def shard_tasks(tasks: list[dict], shard_index: int, num_shards: int) -> list[tuple[int, dict]]:
    return [(index, task) for index, task in enumerate(tasks) if index % num_shards == shard_index]


def shard_path(prop: str, shard_index: int, num_shards: int) -> Path:
    return RUN_DIR / prop / f'shard_{shard_index:02d}_of_{num_shards:02d}.jsonl'


def read_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed = set()
    for line in path.read_text(encoding='utf-8').splitlines():
        try:
            completed.add(json.loads(line)['material_id'])
        except Exception:
            continue
    return completed


def load_expected_route_models(prop: str) -> dict[str, str]:
    """Load route hints only to group evaluation order and cap memory use.

    The route is still recomputed from the frozen knowledge graph for every
    structure, so grouping does not alter the seed-42 sample or its metrics.
    """
    hints = {}
    source_path = original_seed42_path(prop)
    if not source_path.exists():
        return hints
    for line in source_path.read_text(encoding='utf-8').splitlines():
        try:
            row = json.loads(line)
            agent = row.get('agent') or {}
            if prop == 'formation_energy':
                model = FORMATION_ROUTE_BY_CRYSTAL_SYSTEM.get(
                    agent.get('crystal_system')
                )
            else:
                model = agent.get('model')
            if model:
                hints[row['material_id']] = model
        except Exception:
            continue
    return hints


def release_loaded_models(orchestrator) -> None:
    """Release cached weights between model-homogeneous task groups."""
    orchestrator.pool._predictors.clear()
    gc.collect()
    try:
        ctypes.CDLL('libc.so.6').malloc_trim(0)
    except Exception:
        pass


def run_shard(prop: str, shard_index: int, num_shards: int, retries: int,
              route_model: str | None = None,
              material_id: str | None = None) -> None:
    load_private_environment()
    if not os.environ.get('DEEPSEEK_API_KEY'):
        raise SystemExit('DEEPSEEK_API_KEY is not set')

    from graph_orchestrator import GraphOrchestrator

    manifest = load_manifest(prop)
    tasks = shard_tasks(manifest['tasks'], shard_index, num_shards)
    route_hints = load_expected_route_models(prop)
    if route_model:
        tasks = [
            pair for pair in tasks
            if route_hints.get(pair[1]['material_id']) == route_model
        ]
    if material_id:
        tasks = [pair for pair in tasks if pair[1]['material_id'] == material_id]
    if prop == 'formation_energy':
        tasks.sort(key=lambda pair: (route_hints.get(pair[1]['material_id'], ''), pair[0]))
    output_path = shard_path(prop, shard_index, num_shards)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = read_completed(output_path)
    orchestrator = GraphOrchestrator()
    started = time.time()
    active_route_model = None

    for local_index, (global_index, task) in enumerate(tasks, start=1):
        if task['material_id'] in completed:
            continue
        expected_route_model = route_hints.get(task['material_id'])
        if (prop == 'formation_energy' and expected_route_model
                and expected_route_model != active_route_model):
            if active_route_model is not None:
                release_loaded_models(orchestrator)
            active_route_model = expected_route_model
        record = {
            'i': global_index,
            'material_id': task['material_id'],
            'formula': task['formula'],
            'gt': task['gt'],
            'property': prop,
            'unit': UNIT[prop],
            'subset_seed': 42,
            'kg_schema': '2.1',
            'stages': {},
            'errors': [],
        }
        total_start = time.time()
        query = f"Predict the {DISPLAY[prop]} of {task['formula']}."
        record['query'] = query

        stage_start = time.time()
        parsed, attempts, error = retry_call(
            lambda: orchestrator.parse_input(query), retries, 'llm_parse'
        )
        record['stages']['llm_parse'] = {
            'ok': error is None,
            'attempts': attempts,
            'secs': round(time.time() - stage_start, 3),
            'output': parsed,
        }
        if error:
            record['errors'].append(error)

        parsed_targets = parsed.get('target_properties', []) if parsed else []
        target_ok = prop in parsed_targets
        record['stages']['target_check'] = {
            'ok': target_ok,
            'expected': prop,
            'parsed': parsed_targets,
        }
        if parsed is not None and not target_ok:
            record['errors'].append({
                'stage': 'target_check',
                'message': f'Expected {prop}, parsed {parsed_targets}',
            })

        structure = crystal_system = space_group = None
        if target_ok:
            stage_start = time.time()
            try:
                cif_bytes = Path(task['cif_path']).read_bytes()
                structure, crystal_system, space_group = orchestrator.parse_structure(cif_bytes)
                record['stages']['structure_parse'] = {
                    'ok': True,
                    'secs': round(time.time() - stage_start, 3),
                    'crystal_system': crystal_system,
                    'space_group': space_group,
                }
            except Exception as exc:
                record['stages']['structure_parse'] = {
                    'ok': False,
                    'secs': round(time.time() - stage_start, 3),
                }
                record['errors'].append({
                    'stage': 'structure_parse',
                    'message': f'{type(exc).__name__}: {str(exc)}'[:1000],
                })

        plan = None
        if structure is not None:
            stage_start = time.time()
            try:
                plan = orchestrator.kg_plan(crystal_system, [prop])
                step = plan[0] if plan else None
                record['stages']['kg_route'] = {
                    'ok': bool(step and step.get('model') not in (None, 'ensemble')),
                    'secs': round(time.time() - stage_start, 3),
                    'model': step.get('model') if step else None,
                    'mae': step.get('mae') if step else None,
                    'sample_count': step.get('sample_count') if step else None,
                    'route_split': step.get('route_split') if step else None,
                    'route_frozen': step.get('route_frozen') if step else None,
                    'required_tools': step.get('required_tools') if step else None,
                }
                if not record['stages']['kg_route']['ok']:
                    record['errors'].append({'stage': 'kg_route', 'message': 'No executable frozen route'})
            except Exception as exc:
                record['stages']['kg_route'] = {
                    'ok': False,
                    'secs': round(time.time() - stage_start, 3),
                }
                record['errors'].append({
                    'stage': 'kg_route',
                    'message': f'{type(exc).__name__}: {str(exc)}'[:1000],
                })

        results = None
        if plan and record['stages'].get('kg_route', {}).get('ok'):
            stage_start = time.time()
            try:
                results = orchestrator.execute(structure, crystal_system, plan)
                prediction = results.get(prop, {})
                value = prediction.get('value')
                prediction_ok = value is not None and math.isfinite(float(value))
                record['stages']['model_execute'] = {
                    'ok': prediction_ok,
                    'secs': round(time.time() - stage_start, 3),
                    'pred': value,
                    'model': prediction.get('model'),
                    'mae': prediction.get('mae'),
                    'sample_count': prediction.get('sample_count'),
                    'warnings': prediction.get('warnings', []),
                }
                if not prediction_ok:
                    messages = prediction.get('warnings', [])
                    record['errors'].append({
                        'stage': 'model_execute',
                        'message': '; '.join(messages)[:1000] or 'No finite prediction',
                    })
            except Exception as exc:
                record['stages']['model_execute'] = {
                    'ok': False,
                    'secs': round(time.time() - stage_start, 3),
                }
                record['errors'].append({
                    'stage': 'model_execute',
                    'message': f'{type(exc).__name__}: {str(exc)}'[:1000],
                })

        if results and record['stages'].get('model_execute', {}).get('ok'):
            report_formula = (parsed or {}).get('formula') or task['formula']
            stage_start = time.time()
            report, attempts, error = retry_call(
                lambda: orchestrator.generate_report(
                    report_formula, crystal_system, space_group, results
                ),
                retries,
                'report_generation',
            )
            record['stages']['report_generation'] = {
                'ok': error is None and bool(report),
                'attempts': attempts,
                'secs': round(time.time() - stage_start, 3),
                'report': report,
            }
            if error:
                record['errors'].append(error)

        prediction_stage = record['stages'].get('model_execute', {})
        record['pred'] = prediction_stage.get('pred')
        record['model'] = prediction_stage.get('model') or record['stages'].get('kg_route', {}).get('model')
        record['crystal_system'] = crystal_system
        record['prediction_success'] = bool(prediction_stage.get('ok'))
        record['end_to_end_success'] = bool(
            record['stages'].get('llm_parse', {}).get('ok')
            and target_ok
            and record['stages'].get('structure_parse', {}).get('ok')
            and record['stages'].get('kg_route', {}).get('ok')
            and prediction_stage.get('ok')
            and record['stages'].get('report_generation', {}).get('ok')
        )
        record['total_secs'] = round(time.time() - total_start, 3)

        with output_path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')

        if local_index == 1 or local_index % 10 == 0 or local_index == len(tasks):
            elapsed = (time.time() - started) / 60
            print(
                f'[{time.strftime("%H:%M:%S")}] {prop} shard={shard_index}/{num_shards} '
                f'{local_index}/{len(tasks)} id={task["material_id"]} pred={record["pred"]} '
                f'model={record["model"]} e2e={record["end_to_end_success"]} '
                f'elapsed={elapsed:.1f}min',
                flush=True,
            )


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def bootstrap_mae_ci(pairs: list[tuple[float, float]], seed: int, replicates: int = 10_000):
    rng = random.Random(seed)
    n = len(pairs)
    values = []
    for _ in range(replicates):
        total = 0.0
        for _ in range(n):
            pred, gt = pairs[rng.randrange(n)]
            total += abs(pred - gt)
        values.append(total / n)
    return [percentile(values, 0.025), percentile(values, 0.975)]


def merge(prop: str, num_shards: int) -> dict:
    manifest = load_manifest(prop)
    rows = []
    for shard_index in range(num_shards):
        path = shard_path(prop, shard_index, num_shards)
        if not path.exists():
            raise FileNotFoundError(path)
        rows.extend(json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line)
    rows.sort(key=lambda row: row['i'])
    if len(rows) != 1000 or len({row['material_id'] for row in rows}) != 1000:
        raise ValueError(f'Expected 1,000 unique rows for {prop}, got {len(rows)}')
    if [row['material_id'] for row in rows] != [task['material_id'] for task in manifest['tasks']]:
        raise ValueError('Merged output order or membership does not match the frozen manifest')

    valid_lo, valid_hi = VALID_RANGE[prop]
    prediction_rows = [
        row for row in rows
        if row.get('pred') is not None and math.isfinite(float(row['pred']))
    ]
    physically_valid = [
        row for row in prediction_rows if valid_lo <= float(row['pred']) <= valid_hi
    ]
    pairs = [(float(row['pred']), float(row['gt'])) for row in physically_valid]
    mae = sum(abs(pred - gt) for pred, gt in pairs) / len(pairs) if pairs else None
    rmse = math.sqrt(sum((pred - gt) ** 2 for pred, gt in pairs) / len(pairs)) if pairs else None
    stage_failures = Counter()
    for row in rows:
        for error in row.get('errors', []):
            stage_failures[error.get('stage', 'unknown')] += 1

    summary = {
        'property': prop,
        'unit': UNIT[prop],
        'source_csv': str(TEST_CSV),
        'subset_seed': 42,
        'n': len(rows),
        'ordered_material_id_sha256': manifest['ordered_material_id_sha256'],
        'kg_schema': '2.1',
        'route_split': 'routing-construction',
        'prediction_success_n': len(prediction_rows),
        'prediction_coverage': len(prediction_rows) / len(rows),
        'physically_valid_n': len(physically_valid),
        'physically_valid_coverage': len(physically_valid) / len(rows),
        'end_to_end_success_n': sum(bool(row.get('end_to_end_success')) for row in rows),
        'end_to_end_coverage': sum(bool(row.get('end_to_end_success')) for row in rows) / len(rows),
        'mae': mae,
        'rmse': rmse,
        'mae_bootstrap_95_ci': bootstrap_mae_ci(pairs, seed=20260828) if pairs else None,
        'bootstrap_replicates': 10_000,
        'bootstrap_seed': 20260828,
        'route_model_counts': dict(sorted(Counter(row.get('model') for row in rows).items(), key=lambda item: str(item[0]))),
        'crystal_system_counts': dict(sorted(Counter(row.get('crystal_system') for row in rows).items(), key=lambda item: str(item[0]))),
        'failure_stage_counts': dict(stage_failures),
        'timing_seconds': {
            'total': sum(float(row.get('total_secs', 0)) for row in rows),
            'mean_per_structure': statistics.mean(float(row.get('total_secs', 0)) for row in rows),
            'median_per_structure': statistics.median(float(row.get('total_secs', 0)) for row in rows),
        },
        'notes': [
            'MAE and RMSE use finite predictions within the predefined physical range.',
            'End-to-end success additionally requires successful LLM parsing and report generation.',
            'The two properties use separately sampled property-specific seed-42 subsets.',
        ],
    }
    combined_path = RUN_DIR / f'crystalagent_e2e_{prop}_seed42.jsonl'
    with combined_path.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    json_dump(RUN_DIR / f'crystalagent_e2e_{prop}_seed42_summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--property', choices=PROPS)
    parser.add_argument('--prepare-manifests', action='store_true')
    parser.add_argument('--run-shard', action='store_true')
    parser.add_argument('--merge', action='store_true')
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=4)
    parser.add_argument('--retries', type=int, default=2)
    parser.add_argument('--route-model')
    parser.add_argument('--material-id')
    args = parser.parse_args()

    if args.prepare_manifests:
        for prop in PROPS:
            manifest = prepare_manifest(prop)
            print(prop, manifest['n'], manifest['ordered_material_id_sha256'])
        return
    if not args.property:
        parser.error('--property is required for --run-shard and --merge')
    if args.run_shard:
        if not 0 <= args.shard_index < args.num_shards:
            parser.error('--shard-index must be in [0, num-shards)')
        run_shard(
            args.property,
            args.shard_index,
            args.num_shards,
            args.retries,
            route_model=args.route_model,
            material_id=args.material_id,
        )
        return
    if args.merge:
        merge(args.property, args.num_shards)
        return
    parser.error('choose --prepare-manifests, --run-shard or --merge')


if __name__ == '__main__':
    main()
