"""
基准实验脚本：构建晶系×性质×模型 MAE矩阵
用法: python scripts/build_routing_table.py
"""
import os, json, sys
import numpy as np
import torch
from collections import defaultdict
from pymatgen.core import Structure
from tqdm import tqdm

sys.path.insert(0, '/root/autodl-tmp/crystal_agent/backend')

STRUCTURES_DIR  = '/root/autodl-tmp/crystal_agent/data/structures'
TRAIN_META      = '/root/autodl-tmp/crystal_agent/data/routing_table/train_meta.json'
OUTPUT_PATH     = '/root/autodl-tmp/crystal_agent/data/routing_table/mae_matrix.json'
ERRORS_CACHE    = '/root/autodl-tmp/crystal_agent/data/routing_table/errors_cache.json'

CRYSTAL_SYSTEMS = ['三斜', '单斜', '正交', '四方', '三方', '六方', '立方']

MODEL_PROPERTIES = {
    'MEGNet': ['formation_energy', 'band_gap'],
    'M3GNet': ['formation_energy'],
    'ALIGNN': ['formation_energy', 'band_gap'],
    'CGCNN':  ['formation_energy', 'band_gap', 'bulk_modulus', 'shear_modulus'],

    # 新增模型在这里添加，只会补跑缺失的晶系
    # 'CGCNN': ['formation_energy', 'band_gap', 'bulk_modulus', 'shear_modulus'],
}


def load_structure(material_id):
    cif_path = os.path.join(STRUCTURES_DIR, f'{material_id}.cif')
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
    """
    判断哪些模型在哪些晶系上还没跑完
    返回: {model_name: {'props': [...], 'missing_cs': [...]}}
    """
    models_to_run = {}
    for model_name, props in MODEL_PROPERTIES.items():
        missing_cs = set()
        for cs in CRYSTAL_SYSTEMS:
            for prop in props:
                if (cs not in errors
                        or prop not in errors[cs]
                        or model_name not in errors[cs][prop]
                        or len(errors[cs][prop][model_name]) == 0):
                    missing_cs.add(cs)

        if missing_cs:
            models_to_run[model_name] = {
                'props':      props,
                'missing_cs': missing_cs,
            }
            print(f'  {model_name}: 缺失晶系 {sorted(missing_cs)}')
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
                # 过滤 NaN/Inf 再保存
                clean = [e for e in errs if not (np.isnan(e) or np.isinf(e))]
                raw[cs][prop][model] = clean
    with open(ERRORS_CACHE, 'w') as f:
        json.dump(raw, f)


def save_matrix(errors):
    mae_matrix = {}
    for cs in sorted(errors):
        for prop in sorted(errors[cs]):
            key = f'{cs}_{prop}'
            mae_matrix[key] = {'models': {}, 'sample_count': 0}
            for model_name in sorted(errors[cs][prop]):
                errs = errors[cs][prop][model_name]
                if errs:
                    mae_matrix[key]['models'][model_name] = round(float(np.mean(errs)), 4)
                    mae_matrix[key]['sample_count'] = max(
                        mae_matrix[key]['sample_count'], len(errs)
                    )
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(mae_matrix, f, indent=2, ensure_ascii=False)


def load_predictors(models_to_run):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'设备: {device}')
    predictors = {}
    for model_name in models_to_run:
        if model_name == 'MEGNet':
            from models.megnet import MEGNetPredictor
            predictors['MEGNet'] = MEGNetPredictor()
        elif model_name == 'M3GNet':
            from models.m3gnet import M3GNetPredictor
            predictors['M3GNet'] = M3GNetPredictor()
        elif model_name == 'ALIGNN':
            from models.alignn import ALIGNNPredictor
            predictors['ALIGNN'] = ALIGNNPredictor()
        elif model_name == 'CGCNN':
            from models.cgcnn import CGCNNPredictor
            predictors['CGCNN'] = CGCNNPredictor()
        # 新增模型在这里添加
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
        save_matrix(errors)
        return

    print(f'\n加载模型: {list(models_to_run.keys())}')
    predictors = load_predictors(models_to_run)
    print('模型加载完成，开始推理...\n')

    failed = 0
    with tqdm(total=len(train_data), unit='条', ncols=80) as pbar:
        for i, item in enumerate(train_data):
            mid = item['id']
            cs  = item['crystal_system']

            gt = {
                'formation_energy': item.get('formation_energy'),
                'band_gap':         item.get('band_gap'),
                'bulk_modulus':     item.get('bulk_modulus'),
                'shear_modulus':    item.get('shear_modulus'),
            }

            # 判断当前条目是否有任何模型需要跑
            needs_structure = False
            for model_name, info in models_to_run.items():
                if cs in info['missing_cs']:
                    needs_structure = True
                    break

            if not needs_structure:
                pbar.update(1)
                continue

            structure = load_structure(mid)
            if structure is None:
                failed += 1
                pbar.update(1)
                continue

            for model_name, predictor in predictors.items():
                info = models_to_run[model_name]
                # 只跑该模型缺失的晶系
                if cs not in info['missing_cs']:
                    continue
                for prop in info['props']:
                    gt_val = gt.get(prop)
                    if gt_val is None:
                        continue
                    try:
                        result = predictor.predict(structure, prop)
                        err = abs(result['value'] - gt_val)
                        if not np.isnan(err) and not np.isinf(err):
                            errors[cs][prop][model_name].append(err)
                    except Exception:
                        continue

            pbar.update(1)
            pbar.set_postfix({'失败': failed})

            if (i + 1) % 5000 == 0:
                save_errors_cache(errors)
                save_matrix(errors)

    print(f'\n推理完成，失败: {failed} 条')
    save_errors_cache(errors)
    save_matrix(errors)

    print('\n===== 路由矩阵摘要 =====')
    with open(OUTPUT_PATH) as f:
        matrix = json.load(f)
    for key, val in sorted(matrix.items()):
        if val['models']:
            best = min(val['models'], key=val['models'].get)
            print(f'  {key}: 最优={best} MAE={val["models"][best]} (n={val["sample_count"]})')


if __name__ == '__main__':
    main()