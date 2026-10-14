#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM4Mat-Bench MP 测试集基准脚本：CrystalAgent vs LLM 直答
用法:
  python benchmark_llm4mat.py --n 10 --property band_gap --mode both --seed 42
"""
import argparse, csv, json, os, random, re, sys, time
from pathlib import Path

BASE = Path(os.environ.get('CRYSTALAGENT_ROOT', Path(__file__).resolve().parents[1])).resolve()
sys.path.insert(0, str(BASE / 'backend'))
TEST_CSV = BASE / 'LLM4Mat-Bench' / 'test.csv'
STRUCTURES = BASE / 'data' / 'structures'
OUT_DIR = BASE / 'data' / 'benchmark'
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROP_GT = {'band_gap': 'band_gap', 'formation_energy': 'formation_energy_per_atom'}
PROP_UNIT = {'band_gap': 'eV', 'formation_energy': 'eV/atom'}
PROP_DISP = {'band_gap': 'band gap', 'formation_energy': 'formation energy'}


def load_test(property_name, max_n, seed, require_local_cif=True):
    csv.field_size_limit(2 ** 31 - 1)
    gt_col = PROP_GT[property_name]
    rows, seen = [], set()
    with open(TEST_CSV) as f:
        for row in csv.DictReader(f):
            mid = row['material_id']
            if mid in seen:
                continue
            seen.add(mid)
            gt = row[gt_col].strip()
            if not gt:
                continue
            cif_path = os.path.join(STRUCTURES, mid + '.cif')
            if require_local_cif and not os.path.exists(cif_path):
                continue
            try:
                gt_val = float(gt)
            except ValueError:
                continue
            rows.append({
                'material_id': mid,
                'formula': row['formula_pretty'],
                'gt': gt_val,
                'cif_path': cif_path if os.path.exists(cif_path) else None,
            })
    random.seed(seed)
    return random.sample(rows, min(max_n, len(rows)))


def run_llm_direct(client, task, property_name):
    unit = PROP_UNIT[property_name]
    prompt = (f"You are a materials scientist. Predict the {PROP_DISP[property_name]} of "
              f"{task['formula']} in {unit}. Output ONLY a single number, nothing else.")
    try:
        resp = client.chat.completions.create(
            model='deepseek-chat', max_tokens=32,
            messages=[{'role': 'user', 'content': prompt}],
        )
        text = resp.choices[0].message.content.strip()
        m = re.search(r'-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?', text)
        return float(m.group(0)) if m else None
    except Exception as e:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=10)
    ap.add_argument('--property', choices=['band_gap', 'formation_energy'], default='band_gap')
    ap.add_argument('--mode', choices=['agent', 'llm', 'both'], default='both')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--tag', default='')
    args = ap.parse_args()

    tasks = load_test(args.property, args.n, args.seed)
    print(f'[{time.strftime("%H:%M:%S")}] loaded {len(tasks)} tasks (property={args.property})', flush=True)

    out_jsonl = os.path.join(OUT_DIR, f'llm4mat_{args.property}_{args.mode}_{args.seed}.jsonl')
    open(out_jsonl, 'w').close()  # fresh start, avoid mixing with older runs
    results = []

    if args.mode in ('agent', 'both'):
        from graph_orchestrator import GraphOrchestrator
        orch = GraphOrchestrator()
        print(f'[{time.strftime("%H:%M:%S")}] orchestrator ready', flush=True)

    llm_client = None
    if args.mode in ('llm', 'both'):
        from openai import OpenAI
        if not os.environ.get('DEEPSEEK_API_KEY'):
            raise SystemExit('DEEPSEEK_API_KEY not set')
        llm_client = OpenAI(api_key=os.environ.get('DEEPSEEK_API_KEY'), base_url='https://api.deepseek.com')

    for i, task in enumerate(tasks):
        rec = {'i': i, 'material_id': task['material_id'], 'formula': task['formula'], 'gt': task['gt']}
        # ---- Agent 全流程 ----
        if args.mode in ('agent', 'both'):
            t0 = time.time()
            cif_bytes = None
            if task['cif_path']:
                with open(task['cif_path'], 'rb') as f:
                    cif_bytes = f.read()
            user_input = f"Predict the {PROP_DISP[args.property]} of {task['formula']}."
            try:
                res = orch.run(user_input, structure_file=cif_bytes)
                pred = None
                model = None
                if res and res.get('results'):
                    r = res['results'].get(args.property)
                    if r:
                        pred = r.get('value')
                        model = r.get('model')
                rec['agent'] = {'pred': pred, 'model': model, 'secs': round(time.time() - t0, 2),
                                'crystal_system': res.get('crystal_system') if res else None}
                if res and res.get('error'):
                    rec['agent']['error'] = str(res['error'])[:200]
            except Exception as e:
                rec['agent'] = {'pred': None, 'error': str(e)[:200], 'secs': round(time.time() - t0, 2)}
            print(f'[{time.strftime("%H:%M:%S")}] [{i+1}/{len(tasks)}] {task["material_id"]} gt={task["gt"]:.4f} '
                  f'agent={rec["agent"].get("pred")} model={rec["agent"].get("model")}', flush=True)
        # ---- LLM 直答 ----
        if args.mode in ('llm', 'both'):
            t0 = time.time()
            rec['llm'] = {'pred': run_llm_direct(llm_client, task, args.property), 'secs': round(time.time() - t0, 2)}
        results.append(rec)
        with open(out_jsonl, 'a', encoding='utf-8') as f:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    # ---- 汇总 ----
    def summarize(recs, key):
        preds = [(r[key]['pred'], r['gt']) for r in recs if r.get(key) and r[key].get('pred') is not None]
        n = len(recs)
        cov = len(preds)
        if not preds:
            return {'n': n, 'coverage': 0.0, 'mae': None, 'rmse': None}
        mae = sum(abs(p - g) for p, g in preds) / cov
        rmse = (sum((p - g) ** 2 for p, g in preds) / cov) ** 0.5
        return {'n': n, 'coverage': round(cov / n, 4), 'mae': round(mae, 4), 'rmse': round(rmse, 4)}

    summary = {'args': vars(args), 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')}
    if args.mode in ('agent', 'both'):
        summary['agent'] = summarize(results, 'agent')
    if args.mode in ('llm', 'both'):
        summary['llm'] = summarize(results, 'llm')
    out_json = os.path.join(OUT_DIR, f'llm4mat_{args.property}_{args.mode}_{args.seed}_summary.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
