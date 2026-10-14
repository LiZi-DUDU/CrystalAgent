#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run ChemCrow / ChatMOF agents on the same 1000-sample pool (seed 42) as CrystalAgent agent runs.
Usage: run_agent_compare.py --agent chemcrow|chatmof --property band_gap|formation_energy --n 1000 --seed 42 [--limit N] [--workers K]
"""
import argparse, csv, json, os, random, re, sys, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

BASE = Path(os.environ.get('CRYSTALAGENT_ROOT', Path(__file__).resolve().parents[1])).resolve()
sys.path.insert(0, str(BASE / 'backend'))
TEST_CSV = BASE / 'LLM4Mat-Bench' / 'test.csv'
STRUCTURES = BASE / 'data' / 'structures'
OUT_DIR = BASE / 'data' / 'benchmark'
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROP_GT = {'band_gap': 'band_gap', 'formation_energy': 'formation_energy_per_atom'}
PROP_UNIT = {'band_gap': 'eV', 'formation_energy': 'eV/atom'}
PROP_DISP = {'band_gap': 'band gap', 'formation_energy': 'formation energy'}


def load_test(property_name, max_n, seed):
    """Identical filtering + sampling as benchmark_llm4mat.py (require local CIF, seed)."""
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
            if not os.path.exists(cif_path):
                continue
            try:
                gt_val = float(gt)
            except ValueError:
                continue
            rows.append({'material_id': mid, 'formula': row['formula_pretty'], 'gt': gt_val})
    random.seed(seed)
    return random.sample(rows, min(max_n, len(rows)))


def make_agent(agent_name, workers):
    if agent_name == 'chemcrow':
        import langchain
        from chemcrow.agents import chemcrow as cc_module

        def _make_llm(model, temp, api_key, streaming=False):
            return langchain.chat_models.ChatOpenAI(
                temperature=temp, model_name=model, request_timeout=300, streaming=streaming,
                callbacks=[], openai_api_key=api_key,
                openai_api_base='https://api.deepseek.com',
            )
        cc_module._make_llm = _make_llm
        from chemcrow import ChemCrow
        from chemcrow.tools import SMILES2Weight, FuncGroups
        # 仅保留离线 RDKit 工具：默认的 wikipedia/paperscraper/网页工具在本任务不可用或已损坏。
        # 无机化学式无法被 RDKit 解析 -> agent 实际退化为 LLM 知识回答，这正是诚实对比。
        tools = [SMILES2Weight(), FuncGroups()]
        return [ChemCrow(tools=tools, model='deepseek-chat', tools_model='deepseek-chat',
                         temp=0.0, max_iterations=6, verbose=False, streaming=False,
                         openai_api_key=os.environ['DEEPSEEK_API_KEY'])
                for _ in range(workers)]
    else:
        from chatmof import ChatMOF
        from chatmof.llm import get_llm
        llm = get_llm('deepseek-chat', temperature=0.0)
        return [ChatMOF.from_llm(llm=llm, verbose=False, search_internet=False)
                for _ in range(workers)]


def one_call(idx_agent, task, prop, agent_name):
    agent, idx = idx_agent
    prompt = (f"You are a materials scientist. Predict the {PROP_DISP[prop]} of "
              f"{task['formula']} in {PROP_UNIT[prop]}. Output ONLY a single number, nothing else.")
    t0 = time.time()
    try:
        if agent_name == 'chatmof':
            out = agent.run(query=prompt)
        else:
            out = agent.run(prompt)
        text = str(out or '')
        seg = text
        if 'Final Answer' in text:
            seg = text.split('Final Answer', 1)[-1]
        m = re.search(r'-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?', seg)
        return {'material_id': task['material_id'], 'formula': task['formula'], 'gt': task['gt'],
                'pred': float(m.group(0)) if m else None,
                'raw': text[:200], 'secs': round(time.time() - t0, 2)}
    except Exception as e:
        return {'material_id': task['material_id'], 'formula': task['formula'], 'gt': task['gt'],
                'pred': None, 'error': f'{type(e).__name__}: {str(e)[:150]}',
                'secs': round(time.time() - t0, 2)}


def summarize(recs):
    preds = [(r['pred'], r['gt']) for r in recs if r.get('pred') is not None]
    n = len(recs)
    s = {'n': n, 'coverage': round(len(preds) / n, 4) if n else 0}
    if preds:
        s['mae'] = round(sum(abs(p - g) for p, g in preds) / len(preds), 4)
        s['rmse'] = round((sum((p - g) ** 2 for p, g in preds) / len(preds)) ** 0.5, 4)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--agent', choices=['chemcrow', 'chatmof'], required=True)
    ap.add_argument('--property', choices=['band_gap', 'formation_energy'], required=True)
    ap.add_argument('--n', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--workers', type=int, default=1)
    args = ap.parse_args()

    if not os.environ.get('DEEPSEEK_API_KEY'):
        raise SystemExit('DEEPSEEK_API_KEY not set')

    tasks = load_test(args.property, args.n, args.seed)
    if args.limit:
        tasks = tasks[:args.limit]
    print(f'[{time.strftime("%H:%M:%S")}] {args.agent} loaded {len(tasks)} tasks (property={args.property})', flush=True)

    agents = make_agent(args.agent, args.workers)
    out_jsonl = os.path.join(OUT_DIR, f'agentcmp_{args.agent}_{args.property}_seed{args.seed}.jsonl')
    with open(out_jsonl, 'w') as f:
        pass  # fresh start

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        jobs = [((agents[i % args.workers], i), t, args.property, args.agent) for i, t in enumerate(tasks)]
        for i, rec in enumerate(ex.map(lambda j: one_call(j[0], j[1], j[2], j[3]), jobs)):
            rec['i'] = i
            with open(out_jsonl, 'a', encoding='utf-8') as f:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            results.append(rec)
            if (i + 1) % 25 == 0:
                print(f'[{time.strftime("%H:%M:%S")}] {args.agent} {args.property} {i+1}/{len(tasks)} '
                      f'sum_secs={sum(r.get("secs",0) for r in results):.0f}', flush=True)

    s = summarize(results)
    s.update({'agent': args.agent, 'property': args.property, 'n_run': len(tasks),
              'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')})
    json.dump(s, open(os.path.join(OUT_DIR, f'agentcmp_{args.agent}_{args.property}_seed{args.seed}_summary.json'), 'w'),
              ensure_ascii=False, indent=2)
    print(json.dumps(s, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
