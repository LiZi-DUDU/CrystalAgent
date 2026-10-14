"""
基准实验脚本：构建晶系×性质×模型 MAE矩阵
用法: python scripts/build_routing_table.py
"""
import os, json, sys
from pathlib import Path
import numpy as np
import torch
import math
from collections import defaultdict
from pymatgen.core import Structure
from tqdm import tqdm

ROOT = Path(os.environ.get('CRYSTALAGENT_ROOT', Path(__file__).resolve().parents[1])).resolve()
sys.path.insert(0, str(ROOT / 'backend'))

STRUCTURES_DIR = ROOT / 'data' / 'structures'
TRAIN_META = ROOT / 'data' / 'routing_table' / 'train_meta.json'
OUTPUT_PATH = ROOT / 'data' / 'routing_table' / 'mae_matrix.json'
ERRORS_CACHE = ROOT / 'data' / 'routing_table' / 'errors_cache.json'

CRYSTAL_SYSTEMS = ['三斜', '单斜', '正交', '四方', '三方', '六方', '立方']

MODEL_PROPERTIES = {
    'MEGNet': ['formation_energy', 'band_gap'],
    'M3GNet': ['formation_energy'],
    'ALIGNN': ['formation_energy', 'band_gap'],
    'CGCNN':  ['formation_energy', 'band_gap'],
    'CHGNet': ['band_gap'],
    'SevenNet': ['formation_energy'],
    'MACE': ['formation_energy'],
}

ONLY_MODELS = None  # 只跑这些模型，None=全跑


def load_structure(material_id):
    cif_path = STRUCTURES_DIR / f'{material_id}.cif'
    if not os.path.exists(cif_path):
        return None
    try:
        return Structure.from_file(cif_path)
    except Exception:
        return None


def load_errors_cache():
    if os.path.exists(ERRORS_CACHE):
        print('发现误差缓存，加载中...')
        with open(ERRORS_CACHE) as f:
            raw = json.load(f)
        errors = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for cs, props in raw.items():
            for prop, models in props.items():
                for model, errs in models.items():
                    errors[cs][prop][model] = errs
        print('缓存加载完成')
        return errors
    return defaultdict(lambda: defaultdict(lambda: defaultdict(list)))


def get_models_to_run(errors):
    models_to_run = {}
    for model_name, props in MODEL_PROPERTIES.items():
        if ONLY_MODELS and model_name not in ONLY_MODELS:
            continue
        missing = {}
        for cs in CRYSTAL_SYSTEMS:
            missing_props = []
            for prop in props:
                if (cs not in errors
                        or prop not in errors[cs]
                        or model_name not in errors[cs][prop]
                        or len(errors[cs][prop][model_name]) == 0):
                    missing_props.append(prop)
            if missing_props:
                missing[cs] = missing_props
        if missing:
            models_to_run[model_name] = missing
            for cs, mprops in missing.items():
                print(f'  {model_name} / {cs}: 缺失 {mprops}')
        else:
            print(f'  {model_name}: 所有晶系已完成，跳过')
    return models_to_run


def save_errors_cache(errors):
    raw = {}
    for cs, props in errors.items():
        raw[cs] = {}
        for prop, models in props.items():
            raw[cs][prop] = {}
            for model, errs in models.items():
                clean = [e for e in errs if not (np.isnan(e) or np.isinf(e))]
                raw[cs][prop][model] = clean
    with open(ERRORS_CACHE, 'w') as f:
        json.dump(raw, f)


def save_matrix(errors):
    mae_matrix = {}
    # 先加载已有矩阵
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, encoding='utf-8') as f:
            mae_matrix = json.load(f)
    for cs in sorted(errors):
        for prop in sorted(errors[cs]):
            key = f'{cs}_{prop}'
            if key not in mae_matrix:
                mae_matrix[key] = {'models': {}, 'sample_count': 0}
            for model_name in sorted(errors[cs][prop]):
                errs = errors[cs][prop][model_name]
                if errs:
                    mae_matrix[key]['models'][model_name] = round(float(np.mean(errs)), 4)
                    mae_matrix[key]['sample_count'] = max(
                        mae_matrix[key]['sample_count'], len(errs)
                    )
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(mae_matrix, f, indent=2, ensure_ascii=False)


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


def main():
    print('加载 train_meta...')
    with open(TRAIN_META) as f:
        train_data = json.load(f)
    print(f'共 {len(train_data)} 条')

    errors = load_errors_cache()
    print('\n检查模型运行状态:')
    models_to_run = get_models_to_run(errors)

    if not models_to_run:
        print('\n所有模型已跑完，直接生成路由表')
    else:
        predictors = load_all_predictors()
        print(f'\n开始预测 {len(models_to_run)} 个模型...')
        for model_name, missing in models_to_run.items():
            predictor = predictors[model_name]
            props = MODEL_PROPERTIES[model_name]
            for cs in CRYSTAL_SYSTEMS:
                if cs not in missing:
                    continue
                for prop in missing[cs]:
                    print(f'\n  [{model_name}] {cs} / {prop} ...')
                    cs_samples = [item for item in train_data if item.get('crystal_system') == cs]
                    for item in tqdm(cs_samples, desc=f'    {cs}/{prop[:8]}', unit='样本'):
                        mid = item['id']
                        gt_val = item.get(prop)
                        if gt_val is None:
                            continue
                        structure = load_structure(mid)
                        if structure is None:
                            continue
                        try:
                            r = predictor.predict(structure, prop)
                            pred_val = r['value']
                            if not np.isnan(pred_val) and not np.isinf(pred_val):
                                errors[cs][prop][model_name].append(abs(pred_val - gt_val))
                        except Exception as e:
                            continue

        print('\n保存误差缓存...')
        save_errors_cache(errors)

    print('\n生成路由表...')
    save_matrix(errors)

    print('\n===== 路由矩阵摘要 =====')
    with open(OUTPUT_PATH, encoding='utf-8') as f:
        matrix = json.load(f)
    for key in sorted(matrix.keys()):
        entry = matrix[key]
        models = entry.get('models', {})
        if models:
            best_model = min(models, key=models.get)
            print(f'  {key}: 最优={best_model} MAE={models[best_model]} (n={entry.get("sample_count", "?")})')


if __name__ == '__main__':
    main()
