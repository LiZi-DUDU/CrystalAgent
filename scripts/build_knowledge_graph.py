"""
知识图谱构建脚本
基于 NetworkX 构建 SciToolKG
节点：Model、Property、CrystalSystem、Tool
边：PREDICTS、BEST_FOR、PHYSICALLY_IMPLIES、REQUIRES、HAS_COVERAGE
"""
import argparse
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx

# ── 路径配置 ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
MAE_MATRIX_PATH = ROOT / 'data' / 'routing_table' / 'mae_matrix.json'
RESULT_PATH = ROOT / 'data' / 'routing_table' / 'result.json'
KG_OUTPUT_PATH = ROOT / 'data' / 'knowledge_graph' / 'sci_tool_kg.pkl'
KG_JSON_PATH = ROOT / 'data' / 'knowledge_graph' / 'sci_tool_kg.json'

CS_ZH_TO_EN = {
    '三斜': 'triclinic',
    '单斜': 'monoclinic',
    '正交': 'orthorhombic',
    '四方': 'tetragonal',
    '三方': 'trigonal',
    '六方': 'hexagonal',
    '立方': 'cubic',
}

# ── 节点数据定义 ──────────────────────────────────────────

MODELS = {
    'MEGNet': {
        'type':           'Model',
        'architecture':   'GNN',
        'training_data':  'Materials Project',
        'speed':          'fast',
        'supported_props': ['formation_energy', 'band_gap'],
        'element_coverage': 'broad',   # 覆盖大多数元素
        'is_available':   True,
        'paper':          'Chen et al. 2019',
        'notes':          '引入全局状态向量，同时支持晶体和分子',
    },
    'M3GNet': {
        'type':           'Model',
        'architecture':   'GNN',
        'training_data':  'Materials Project',
        'speed':          'fast',
        'supported_props': ['formation_energy'],
        'element_coverage': 'broad',
        'is_available':   True,
        'paper':          'Chen et al. 2022',
        'notes':          '势能面模型，形成能预测精度高',
    },
    'ALIGNN': {
        'type':           'Model',
        'architecture':   'GNN',
        'training_data':  'JARVIS-DFT',
        'speed':          'fast',
        'supported_props': ['formation_energy', 'band_gap'],
        'element_coverage': 'broad',
        'is_available':   True,
        'paper':          'Choudhary & DeCost 2021',
        'notes':          '显式编码键角信息，带隙预测综合最优',
    },
    'CHGNet': {
        'type':           'Model',
        'architecture':   'Charge-aware GNN',
        'training_data':  'Materials Project MPtrj',
        'speed':          'fast',
        'supported_props': ['band_gap'],
        'element_coverage': 'broad',
        'is_available':   True,
        'paper':          'Deng et al. 2023',
        'notes':          '电荷感知GNN，冻结骨干+MLP带隙预测头',
    },
    'SevenNet': {
        'type':           'Model',
        'architecture':   'E(3)-equivariant',
        'training_data':  'MP + OMat24',
        'speed':          'fast',
        'supported_props': ['formation_energy'],
        'element_coverage': 'broad',
        'is_available':   True,
        'paper':          'Park et al. 2024',
        'notes':          '通用等变势，需元素参考能换算得到形成能',
    },
    'MACE': {
        'type':           'Model',
        'architecture':   'Many-body ACE',
        'training_data':  'Materials Project MPtrj',
        'speed':          'medium',
        'supported_props': ['formation_energy'],
        'element_coverage': 'broad',
        'is_available':   True,
        'paper':          'Batatia et al. 2024',
        'notes':          '通用多体势，需元素参考能换算得到形成能',
    },
    'CGCNN': {
        'type':           'Model',
        'architecture':   'GNN',
        'training_data':  'Materials Project',
        'speed':          'slow',   # 每次需要IO操作
        'supported_props': ['formation_energy', 'band_gap', 'bulk_modulus', 'shear_modulus'],
        'element_coverage': 'broad',
        'is_available':   True,
        'paper':          'Xie & Grossman 2018',
        'notes':          '经典模型，支持性质种类最多',
    },
}

PROPERTIES = {
    'formation_energy': {
        'type':         'Property',
        'unit':         'eV/atom',
        'valid_range':  (-10.0, 5.0),   # 物理合理范围
        'description':  '每原子形成能（相对于元素参考态）；负值仅表示相对于元素形成放热，不等同于对所有竞争相热力学稳定',
        'elemental_reference_threshold': 0.0,
    },
    'band_gap': {
        'type':         'Property',
        'unit':         'eV',
        'valid_range':  (0.0, 20.0),    # 带隙不能为负
        'description':  '预测电子带隙；接近零的预测与金属或半金属行为相符，但也可能受模型与泛函误差影响',
        'near_zero_threshold': 0.1,
    },
    'bulk_modulus': {
        'type':         'Property',
        'unit':         'log(GPa)',
        'valid_range':  (-1.0, 4.0),    # log10(0.1GPa) ~ log10(10000GPa)
        'description':  '体模量（Voigt-Reuss-Hill平均），衡量材料抗压缩能力',
    },
    'shear_modulus': {
        'type':         'Property',
        'unit':         'log(GPa)',
        'valid_range':  (-1.0, 4.0),
        'description':  '剪切模量（Voigt-Reuss-Hill平均），衡量材料抗剪切能力',
    },
    'refractive_index': {
        'type':         'Property',
        'unit':         'unitless',
        'valid_range':  (1.0, 5.0),
        'description':  'Optical refractive index n, related to band gap and dielectric constant',
    },
    'dielectric_constant': {
        'type':         'Property',
        'unit':         'unitless',
        'valid_range':  (1.0, 100.0),
        'description':  'Static dielectric constant; wide-gap materials tend to have lower epsilon',
    },
    'density': {
        'type':         'Property',
        'unit':         'g/cm3',
        'valid_range':  (0.5, 25.0),
        'description':  'Theoretical crystal density; high density implies large bulk modulus (Birch law)',
    },
    'thermodynamic_stability': {
        'type':         'Property',
        'unit':         'meV/atom',
        'valid_range':  (0.0, 500.0),
        'description':  'Energy above convex hull; <100 meV/atom considered stable',
        'stability_threshold': 100.0,
    },
    'synthesis_feasibility': {
        'type':         'Property',
        'unit':         'score',
        'valid_range':  (0.0, 1.0),
        'description':  'Predicted synthesizability score combining thermodynamics and literature',
    },
    'speed_of_sound': {
        'type':         'Property',
        'unit':         'm/s',
        'valid_range':  (100.0, 20000.0),
        'description':  'Longitudinal speed of sound, derived from bulk modulus + shear modulus + density',
    },

    'poisson_ratio': {
        'type':         'Property',
        'unit':         'dimensionless',
        'valid_range':  (-1.0, 0.5),    # 泊松比物理约束
        'description':  '泊松比，描述横向应变与轴向应变之比',
    },
    'fermi_energy': {
        'type':         'Property',
        'unit':         'eV',
        'valid_range':  (-20.0, 20.0),
        'description':  '费米能级',
    },
}

CRYSTAL_SYSTEMS = {
    '三斜': {
        'type':             'CrystalSystem',
        'spacegroup_range': (1, 2),
        'symmetry_level':   'lowest',
        'typical_materials': ['CuAl2O4变体', '矿物feldspar'],
        'train_count':      16898,   # 从mae_matrix读取
    },
    '单斜': {
        'type':             'CrystalSystem',
        'spacegroup_range': (3, 15),
        'symmetry_level':   'low',
        'typical_materials': ['单斜ZrO2', 'VO2低温相'],
        'train_count':      25439,
    },
    '正交': {
        'type':             'CrystalSystem',
        'spacegroup_range': (16, 74),
        'symmetry_level':   'medium',
        'typical_materials': ['正交BaTiO3', 'GaN正交相'],
        'train_count':      22715,
    },
    '四方': {
        'type':             'CrystalSystem',
        'spacegroup_range': (75, 142),
        'symmetry_level':   'medium',
        'typical_materials': ['四方BaTiO3', 'TiO2金红石'],
        'train_count':      11766,
    },
    '三方': {
        'type':             'CrystalSystem',
        'spacegroup_range': (143, 167),
        'symmetry_level':   'medium_high',
        'typical_materials': ['α-Al2O3', 'α-Fe2O3'],
        'train_count':      9099,
    },
    '六方': {
        'type':             'CrystalSystem',
        'spacegroup_range': (168, 194),
        'symmetry_level':   'high',
        'typical_materials': ['GaN纤锌矿', 'MoS2 2H相'],
        'train_count':      7474,
    },
    '立方': {
        'type':             'CrystalSystem',
        'spacegroup_range': (195, 230),
        'symmetry_level':   'highest',
        'typical_materials': ['NaCl', 'Si', 'Cu'],
        'train_count':      14668,
    },
}

TOOLS = {
    'spglib': {
        'type':        'Tool',
        'function':    '从结构文件确定空间群和晶系',
        'input':       'Structure',
        'output':      'crystal_system, space_group',
        'is_available': True,
    },
    'CifParser': {
        'type':        'Tool',
        'function':    '解析CIF/POSCAR格式的结构文件',
        'input':       'file_path',
        'output':      'Structure',
        'is_available': True,
    },
    'LLM_parser': {
        'type':        'Tool',
        'function':    '从自然语言提取化学式和目标性质',
        'input':       'natural_language',
        'output':      'formula, target_properties',
        'is_available': True,
    },
    'polymorph_checker': {
        'type':        'Tool',
        'function':    '检测化学式是否存在多态性',
        'input':       'formula',
        'output':      'polymorphs_list',
        'is_available': True,
    },
    'FeatureExtractor': {
        'type':        'Tool',
        'function':    'Extract atomic, bonding, and structural features for GNN input',
        'input':       'Structure',
        'output':      'graph_features',
        'is_available': True,
        'category':    'feature_engineering',
    },
    'StructureOptimizer': {
        'type':        'Tool',
        'function':    'Optimize atomic positions and cell parameters before prediction',
        'input':       'Structure',
        'output':      'Structure_optimized',
        'is_available': False,
        'category':    'preprocessing',
    },
    'StructureVisualizer': {
        'type':        'Tool',
        'function':    'Generate 3D crystal structure visualization for reports',
        'input':       'Structure',
        'output':      'image',
        'is_available': False,
        'category':    'visualization',
    },
    'EnsembleAggregator': {
        'type':        'Tool',
        'function':    'Aggregate multi-model predictions with inverse-MAE weighting',
        'input':       'predictions_list',
        'output':      'ensemble_prediction',
        'is_available': True,
        'category':    'postprocessing',
    },
    'ConvexHull': {
        'type':        'Tool',
        'function':    'Compute energy above convex hull for thermodynamic stability assessment',
        'input':       'formation_energy, composition',
        'output':      'energy_above_hull',
        'is_available': False,
        'category':    'analysis',
        'notes':       'Requires Materials Project API for phase diagram data',
    },
    'UncertaintyEstimator': {
        'type':        'Tool',
        'function':    'Estimate prediction uncertainty from model disagreement and training coverage',
        'input':       'predictions, model_info',
        'output':      'uncertainty_score',
        'is_available': True,
        'category':    'postprocessing',
        'notes':       'PBE-level band gap predictions systematically underestimate by 30-50%',
    },
    'MLP_BandgapHead': {
        'type':          'Tool',
        'function':      'Apply the supervised regression head to frozen CHGNet representations',
        'input':         'CHGNet representation',
        'output':        'band_gap',
        'is_available':  True,
        'category':      'model_head',
    },
    'ElementReference': {
        'type':          'Tool',
        'function':      'Provide elemental reference energies fitted on the routing-construction split',
        'input':         'composition',
        'output':        'elemental_reference_energy',
        'is_available':  True,
        'category':      'energy_reference',
    },
    'EnergyConverter': {
        'type':          'Tool',
        'function':      'Convert model total energy to per-atom formation energy using frozen elemental references',
        'input':         'total_energy, composition, elemental_reference_energy',
        'output':        'formation_energy',
        'is_available':  True,
        'category':      'postprocessing',
    },

}

# ── 性质间的物理含义关系 ──────────────────────────────────
PHYSICAL_IMPLICATIONS = [
    {
        'from':      'formation_energy',
        'to':        'band_gap',
        'condition': 'formation_energy > 0',
        'implication': '正形成能表示相对于所用元素参考态形成吸热；不能仅据此判断对竞争相的稳定性或带隙结果是否有效',
        'action':    'warn',
    },
    {
        'from':      'bulk_modulus',
        'to':        'shear_modulus',
        'condition': 'always',
        'implication': '体模量和剪切模量通常一起预测，用于计算泊松比',
        'action':    'suggest',
    },
    {
        'from':      'band_gap',
        'to':        'dielectric_constant',
        'condition': 'band_gap > 0.5',
        'implication': 'Band gap inversely correlates with dielectric constant; wide-gap materials have lower epsilon',
        'action':    'note',
    },
    {
        'from':      'band_gap',
        'to':        'synthesis_feasibility',
        'condition': 'band_gap < 0',
        'implication': 'Negative (metallic) band gap may indicate model failure for this structure type',
        'action':    'warn',
    },
    {
        'from':      'formation_energy',
        'to':        'thermodynamic_stability',
        'condition': 'formation_energy < 0',
        'implication': '负形成能表示相对于元素参考态形成放热，但不能代替凸包能或有限温度自由能来判定相稳定性',
        'action':    'note',
    },
    {
        'from':      'formation_energy',
        'to':        'synthesis_feasibility',
        'condition': 'formation_energy > 0',
        'implication': '正形成能相对于元素参考态不利；不能仅由该值推断可合成性或具体合成路径',
        'action':    'warn',
    },
    {
        'from':      'thermodynamic_stability',
        'to':        'synthesis_feasibility',
        'condition': 'thermodynamic_stability > 100',
        'implication': 'Energy above hull >100 meV/atom: material likely unstable and difficult to synthesize',
        'action':    'warn',
    },
    {
        'from':      'density',
        'to':        'bulk_modulus',
        'condition': 'density > 5',
        'implication': 'High density materials tend to have large bulk modulus (Birch law)',
        'action':    'prioritize',
    },
    {
        'from':      'bulk_modulus',
        'to':        'speed_of_sound',
        'condition': 'always',
        'implication': 'Bulk and shear moduli + density determine acoustic wave speeds',
        'action':    'derive',
        'formula':   'v_p = sqrt((B + 4G/3) / rho)',
    },
    {
        'from':      'shear_modulus',
        'to':        'speed_of_sound',
        'condition': 'always',
        'implication': 'Shear modulus determines transverse sound velocity',
        'action':    'derive',
        'formula':   'v_s = sqrt(G / rho)',
    },
]


def _split_route_key(key: str) -> tuple[str | None, str | None]:
    """Parse ``{Chinese crystal system}_{property}`` without splitting property underscores."""
    for cs_name in CRYSTAL_SYSTEMS:
        prefix = cs_name + '_'
        if key.startswith(prefix):
            return cs_name, key[len(prefix):]
    return None, None


def _validate_frozen_route(cs: str, prop: str, best_model: str,
                           frozen_routes: dict | None) -> None:
    if not frozen_routes:
        return
    expected = frozen_routes.get(prop, {}).get(CS_ZH_TO_EN[cs])
    if expected is not None and expected != best_model:
        raise ValueError(
            f'Route mismatch for {cs}/{prop}: matrix selects {best_model}, '
            f'but result.json freezes {expected}'
        )


def build_graph(mae_matrix: dict, frozen_routes: dict | None = None) -> nx.MultiDiGraph:
    # Multiple semantic relations can connect the same crystal-system/model pair.
    # A DiGraph would overwrite BEST_FOR with HAS_COVERAGE.
    G = nx.MultiDiGraph(
        schema_version='2.1',
        route_split='routing-construction',
        route_matrix=MAE_MATRIX_PATH,
        route_manifest=RESULT_PATH,
        built_at_utc=datetime.now(timezone.utc).isoformat(),
    )

    # ── 添加节点 ──────────────────────────────────────────
    for name, attrs in MODELS.items():
        G.add_node(f'Model:{name}', **attrs, label=name)

    for name, attrs in PROPERTIES.items():
        G.add_node(f'Property:{name}', **attrs, label=name)

    for name, attrs in CRYSTAL_SYSTEMS.items():
        node_attrs = dict(attrs)
        route_counts = [
            entry.get('sample_count', 0)
            for key, entry in mae_matrix.items()
            if key.startswith(name + '_')
        ]
        if route_counts:
            node_attrs['train_count'] = max(route_counts)
        node_attrs['count_split'] = 'routing-construction'
        G.add_node(f'CrystalSystem:{name}', **node_attrs, label=name)

    for name, attrs in TOOLS.items():
        G.add_node(f'Tool:{name}', **attrs, label=name)

    # ── 添加边：Model PREDICTS Property ──────────────────
    for model_name, model_data in MODELS.items():
        for prop in model_data['supported_props']:
            if f'Property:{prop}' in G.nodes:
                G.add_edge(
                    f'Model:{model_name}',
                    f'Property:{prop}',
                    relation='PREDICTS',
                )

    # ── 添加边：从MAE矩阵提取 CrystalSystem BEST_FOR Model ──
    for key, entry in mae_matrix.items():
        cs, prop = _split_route_key(key)

        if not cs or not prop:
            continue

        models_mae = entry.get('models', {})
        sample_count = entry.get('sample_count', 0)

        if not models_mae:
            continue

        # 最优模型
        best_model = min(models_mae, key=models_mae.get)
        best_mae   = models_mae[best_model]
        _validate_frozen_route(cs, prop, best_model, frozen_routes)

        # CrystalSystem -> Model (BEST_FOR)
        if f'CrystalSystem:{cs}' in G.nodes and f'Model:{best_model}' in G.nodes:
            G.add_edge(
                f'CrystalSystem:{cs}',
                f'Model:{best_model}',
                key        = f'BEST_FOR:{prop}',
                relation   = 'BEST_FOR',
                property   = prop,
                mae        = best_mae,
                sample_count = sample_count,
                split      = 'routing-construction',
                frozen     = True,
                source     = RESULT_PATH,
            )

        # 所有模型的MAE数据也加进去
        for model_name, mae in models_mae.items():
            if f'Model:{model_name}' in G.nodes:
                G.add_edge(
                    f'CrystalSystem:{cs}',
                    f'Model:{model_name}',
                    key          = f'HAS_COVERAGE:{prop}',
                    relation     = 'HAS_COVERAGE',
                    property     = prop,
                    mae          = mae,
                    sample_count = sample_count,
                    split        = 'routing-construction',
                    source       = MAE_MATRIX_PATH,
                )

    # ── 添加边：Property PHYSICALLY_IMPLIES Property ──────
    for impl in PHYSICAL_IMPLICATIONS:
        src = f'Property:{impl["from"]}'
        dst = f'Property:{impl["to"]}'
        if src in G.nodes and dst in G.nodes:
            G.add_edge(
                src, dst,
                relation   = 'PHYSICALLY_IMPLIES',
                condition  = impl['condition'],
                implication = impl['implication'],
                action     = impl['action'],
            )

    # ── 添加边：Model REQUIRES Tool ───────────────────────
    model_tool_map = {
        'MEGNet':   ['CifParser', 'spglib', 'FeatureExtractor'],
        'M3GNet':   ['CifParser', 'spglib', 'FeatureExtractor'],
        'ALIGNN':   ['CifParser', 'spglib', 'FeatureExtractor'],
        'CGCNN':    ['CifParser', 'spglib', 'FeatureExtractor'],
        'CHGNet':   ['CifParser', 'spglib', 'FeatureExtractor', 'MLP_BandgapHead'],
        'SevenNet': ['CifParser', 'spglib', 'ElementReference', 'EnergyConverter'],
        'MACE':     ['CifParser', 'spglib', 'ElementReference', 'EnergyConverter'],
    }
    for model_name, tools in model_tool_map.items():
        for tool in tools:
            if f'Model:{model_name}' in G.nodes and f'Tool:{tool}' in G.nodes:
                G.add_edge(
                    f'Model:{model_name}',
                    f'Tool:{tool}',
                    relation='REQUIRES',
                )

    # ── 添加边：LLM_parser -> Property（可以提取的目标性质）
    for prop in PROPERTIES:
        G.add_edge(
            'Tool:LLM_parser',
            f'Property:{prop}',
            relation='CAN_IDENTIFY',
        )

    return G


def print_graph_summary(G: nx.DiGraph):
    print('\n===== 知识图谱摘要 =====')
    print(f'节点总数: {G.number_of_nodes()}')
    print(f'边总数:   {G.number_of_edges()}')

    # 按类型统计节点
    type_counts = {}
    for node, data in G.nodes(data=True):
        t = data.get('type', 'Unknown')
        type_counts[t] = type_counts.get(t, 0) + 1
    print('\n节点类型分布:')
    for t, count in sorted(type_counts.items()):
        print(f'  {t}: {count}')

    # 按类型统计边
    rel_counts = {}
    for u, v, data in G.edges(data=True):
        r = data.get('relation', 'Unknown')
        rel_counts[r] = rel_counts.get(r, 0) + 1
    print('\n边类型分布:')
    for r, count in sorted(rel_counts.items()):
        print(f'  {r}: {count}')

    # 打印每个晶系的最优模型
    print('\n各晶系×性质最优模型:')
    best_for_edges = [
        (u, v, d) for u, v, d in G.edges(data=True)
        if d.get('relation') == 'BEST_FOR'
    ]
    for u, v, d in sorted(best_for_edges, key=lambda x: x[0]):
        cs    = u.replace('CrystalSystem:', '')
        model = v.replace('Model:', '')
        print(f'  {cs} / {d["property"]}: {model} (MAE={d["mae"]})')


def save_graph(G: nx.MultiDiGraph, pkl_path: str, json_path: str):
    # 保存为 pickle（用于程序加载）
    Path(pkl_path).parent.mkdir(parents=True, exist_ok=True)
    with open(pkl_path, 'wb') as f:
        pickle.dump(G, f)
    print(f'\n图谱已保存至 {pkl_path}')

    # 保存为 JSON（可读版本）
    graph_json = {
        'nodes': [
            {'id': node, **{k: v for k, v in data.items() if isinstance(v, (str, int, float, bool, list))}}
            for node, data in G.nodes(data=True)
        ],
        'graph': {
            k: v for k, v in G.graph.items()
            if isinstance(v, (str, int, float, bool, list))
        },
        'edges': [
            {
                'source': u,
                'target': v,
                'key': edge_key,
                **{k: v2 for k, v2 in data.items() if isinstance(v2, (str, int, float, bool))},
            }
            for u, v, edge_key, data in G.edges(keys=True, data=True)
        ],
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(graph_json, f, indent=2, ensure_ascii=False)
    print(f'可读版本已保存至 {json_path}')


def main():
    parser = argparse.ArgumentParser(description='Build the CrystalAgent scientific-tool knowledge graph.')
    parser.add_argument('--matrix', default=MAE_MATRIX_PATH)
    parser.add_argument('--result', default=RESULT_PATH)
    parser.add_argument('--pkl-out', default=KG_OUTPUT_PATH)
    parser.add_argument('--json-out', default=KG_JSON_PATH)
    args = parser.parse_args()

    print('加载 MAE 矩阵...')
    with open(args.matrix) as f:
        mae_matrix = json.load(f)
    print(f'共 {len(mae_matrix)} 条路由记录')

    with open(args.result) as f:
        result = json.load(f)
    frozen_routes = result.get('metadata', {}).get('frozen_routes', {})

    print('构建知识图谱...')
    G = build_graph(mae_matrix, frozen_routes=frozen_routes)

    print_graph_summary(G)
    save_graph(G, args.pkl_out, args.json_out)


if __name__ == '__main__':
    main()
