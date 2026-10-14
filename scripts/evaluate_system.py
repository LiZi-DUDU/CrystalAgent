"""
系统评估脚本
对比：路由器 vs 单模型baseline vs 无路由集成baseline
按晶系分层报告结果
"""
import os, json, sys
from pathlib import Path
import numpy as np
import math
from collections import defaultdict
from pymatgen.core import Structure
from tqdm import tqdm

ROOT = Path(os.environ.get('CRYSTALAGENT_ROOT', Path(__file__).resolve().parents[1])).resolve()
sys.path.insert(0, str(ROOT / 'backend'))

STRUCTURES_DIR = ROOT / 'data' / 'structures'
EVAL_META = ROOT / 'data' / 'routing_table' / 'eval_meta.json'
MATRIX_PATH = ROOT / 'data' / 'routing_table' / 'mae_matrix.json'
OUTPUT_PATH = ROOT / 'data' / 'routing_table' / 'eval_results.json'

# 单模型baseline：综合表现最好的单一模型
SINGLE_BASELINE = 'ALIGNN'

# 各模型支持的性质
MODEL_PROPERTIES = {
    'MEGNet': ['formation_energy', 'band_gap'],
    'M3GNet': ['formation_energy'],
    'ALIGNN': ['formation_energy', 'band_gap'],
    'CGCNN':  ['formation_energy', 'band_gap'],
    'CHGNet': ['band_gap'],
    'SevenNet': ['formation_energy'],
    'MACE': ['formation_energy'],
}

TARGET_PROPERTIES = ['formation_energy', 'band_gap']


def load_structure(material_id):
    cif_path = STRUCTURES_DIR / f'{material_id}.cif'
    if not os.path.exists(cif_path):
        return None
    try:
        return Structure.from_file(cif_path)
    except Exception:
        return None


def load_router(matrix_path):
    with open(matrix_path) as f:
        return json.load(f)


def route(matrix, cs, prop):
    key = f'{cs}_{prop}'
    entry = matrix.get(key, {})
    models = entry.get('models', {})
    if not models:
        return None
    return min(models, key=models.get)


def load_all_predictors():
    predictors = {}
    from models.megnet import MEGNetPredictor
    predictors['MEGNet'] = MEGNetPredictor()
    from models.m3gnet import M3GNetPredictor
    predictors['M3GNet'] = M3GNetPredictor()
    from models.alignn import ALIGNNPredictor
    predictors['ALIGNN'] = ALIGNNPredictor()
    from models.cgcnn import CGCNNPredictor
    from models.chgnet import CHGNetPredictor
    predictors['CHGNet'] = CHGNetPredictor()
    predictors['CGCNN'] = CGCNNPredictor()
    from models.sevennet import SevenNetPredictor
    predictors['SevenNet'] = SevenNetPredictor()
    from models.mace import MACEPredictor
    predictors['MACE'] = MACEPredictor()
    return predictors


def get_gt(item, prop):
    val = item.get(prop)
    if val is None:
        return None
    # 体模量和剪切模量转 log10
    if prop in ('bulk_modulus', 'shear_modulus'):
        if val <= 0:
            return None
        return math.log10(val)
    return val


def main():
    print('加载评估集...')
    with open(EVAL_META) as f:
        eval_data = json.load(f)
    print(f'共 {len(eval_data)} 条')

    print('加载路由表...')
    matrix = load_router(MATRIX_PATH)

    print('加载模型...')
    predictors = load_all_predictors()
    print('模型加载完成\n')

    # 收集误差
    # errors[method][prop][cs] = [误差列表]
    # method: 'router', 'single_best', 'ensemble'
    errors      = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    route_correct = defaultdict(list)  # 路由准确率

    failed = 0
    with tqdm(total=len(eval_data), unit='条', ncols=80) as pbar:
        for item in eval_data:
            mid = item['id']
            cs  = item['crystal_system']

            structure = load_structure(mid)
            if structure is None:
                failed += 1
                pbar.update(1)
                continue

            for prop in TARGET_PROPERTIES:
                gt_val = get_gt(item, prop)
                if gt_val is None:
                    continue

                # 收集所有模型的预测值
                preds = {}
                for model_name, predictor in predictors.items():
                    if prop not in MODEL_PROPERTIES[model_name]:
                        continue
                    try:
                        r = predictor.predict(structure, prop)
                        pred_val = r['value']
                        if prop in ('bulk_modulus', 'shear_modulus') and model_name != 'CGCNN':
                            continue  # 只有CGCNN支持这两个性质
                        if not np.isnan(pred_val) and not np.isinf(pred_val):
                            preds[model_name] = pred_val
                    except Exception:
                        continue

                if not preds:
                    continue

                # 1. 路由器预测
                routed_model = route(matrix, cs, prop)
                if routed_model and routed_model in preds:
                    err = abs(preds[routed_model] - gt_val)
                    errors['router'][prop][cs].append(err)

                    # 路由准确率：路由选择是否是误差最小的模型
                    best_model = min(preds, key=lambda m: abs(preds[m] - gt_val))
                    route_correct[prop].append(routed_model == best_model)

                # 2. 单模型baseline
                if SINGLE_BASELINE in preds:
                    err = abs(preds[SINGLE_BASELINE] - gt_val)
                    errors['single_best'][prop][cs].append(err)

                # 3. 无路由集成baseline（等权平均）
                if len(preds) > 1:
                    ensemble_pred = np.mean(list(preds.values()))
                    err = abs(ensemble_pred - gt_val)
                    errors['ensemble'][prop][cs].append(err)

            pbar.update(1)
            pbar.set_postfix({'失败': failed})

    print(f'\n评估完成，失败: {failed} 条')

    # 输出结果
    print('\n' + '='*60)
    print('系统评估结果')
    print('='*60)

    for prop in TARGET_PROPERTIES:
        print(f'\n【{prop}】')
        print(f'  {"方法":<15} {"整体MAE":>10} {"路由准确率":>12}')
        print(f'  {"-"*40}')

        for method in ['router', 'single_best', 'ensemble']:
            all_errs = []
            for cs in errors[method][prop]:
                all_errs.extend(errors[method][prop][cs])
            if not all_errs:
                continue
            overall_mae = np.mean(all_errs)
            acc = np.mean(route_correct[prop]) if method == 'router' and route_correct[prop] else '-'
            acc_str = f'{acc:.2%}' if isinstance(acc, float) else acc
            print(f'  {method:<15} {overall_mae:>10.4f} {acc_str:>12}')

        print(f'\n  分晶系MAE（router vs single_best）:')
        for cs in sorted(errors['router'][prop]):
            r_mae  = np.mean(errors['router'][prop][cs]) if errors['router'][prop][cs] else None
            s_mae  = np.mean(errors['single_best'][prop][cs]) if errors['single_best'][prop][cs] else None
            routed = route(matrix, cs, prop)
            if r_mae and s_mae:
                diff = ((r_mae - s_mae) / s_mae) * 100
                flag = '↑' if diff > 0 else '↓'
                print(f'    {cs}: router={r_mae:.4f} single={s_mae:.4f} {flag}{abs(diff):.1f}% (路由→{routed})')

    # 保存结果
    results = {
        'overall': {},
        'by_crystal_system': {},
        'routing_accuracy': {}
    }
    for prop in TARGET_PROPERTIES:
        results['overall'][prop] = {}
        results['by_crystal_system'][prop] = {}
        for method in ['router', 'single_best', 'ensemble']:
            all_errs = [e for cs in errors[method][prop] for e in errors[method][prop][cs]]
            results['overall'][prop][method] = round(float(np.mean(all_errs)), 4) if all_errs else None
        results['routing_accuracy'][prop] = round(float(np.mean(route_correct[prop])), 4) if route_correct[prop] else None
        for cs in sorted(errors['router'][prop]):
            results['by_crystal_system'][prop][cs] = {
                'router':      round(float(np.mean(errors['router'][prop][cs])), 4),
                'single_best': round(float(np.mean(errors['single_best'][prop][cs])), 4) if errors['single_best'][prop][cs] else None,
            }

    with OUTPUT_PATH.open('w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'\n结果已保存至 {OUTPUT_PATH}')


if __name__ == '__main__':
    main()
