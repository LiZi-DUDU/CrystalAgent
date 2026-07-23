"""
知识图谱构建脚本
基于 NetworkX 构建 SciToolKG
节点：Model、Property、CrystalSystem、Tool
边：PREDICTS、BEST_FOR、PHYSICALLY_IMPLIES、REQUIRES、HAS_COVERAGE
"""
import json
import pickle
import networkx as nx
from pathlib import Path

# ── 路径配置 ──────────────────────────────────────────────
MAE_MATRIX_PATH = 'data/routing_table/mae_matrix.json'
KG_OUTPUT_PATH  = 'data/knowledge_graph/sci_tool_kg.pkl'
KG_JSON_PATH    = 'data/knowledge_graph/sci_tool_kg.json'  # 可读版本

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
    'CHGNet': {
        'type':           'Model',
        'architecture':   'GNN',
        'training_data':  'Materials Project Trajectory',
        'speed':          'fast',
        'supported_props': [],   # 暂不参与性质预测路由
        'element_coverage': 'broad',
        'is_available':   False,
        'paper':          'Deng et al. 2023',
        'notes':          '势能面模型，适合分子动力学，暂不用于性质预测',
    },
}

PROPERTIES = {
    'formation_energy': {
        'type':         'Property',
        'unit':         'eV/atom',
        'valid_range':  (-10.0, 5.0),   # 物理合理范围
        'description':  '每原子形成能，负值表示材料热力学稳定',
        'stability_threshold': 0.0,     # >0 说明不稳定
    },
    'band_gap': {
        'type':         'Property',
        'unit':         'eV',
        'valid_range':  (0.0, 20.0),    # 带隙不能为负
        'description':  '电子带隙，0为金属，>0为半导体/绝缘体',
        'metal_threshold': 0.1,         # <0.1 eV 视为金属
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
}

# ── 性质间的物理含义关系 ──────────────────────────────────
PHYSICAL_IMPLICATIONS = [
    {
        'from':      'formation_energy',
        'to':        'band_gap',
        'condition': 'formation_energy > 0',
        'implication': '形成能为正说明材料热力学不稳定，带隙预测参考价值低',
        'action':    'warn',   # warn/skip
    },
    {
        'from':      'band_gap',
        'to':        'shear_modulus',
        'condition': 'band_gap < 0.1',
        'implication': '带隙接近0说明是金属，剪切模量预测更有实际意义',
        'action':    'prioritize',
    },
    {
        'from':      'bulk_modulus',
        'to':        'shear_modulus',
        'condition': 'always',
        'implication': '体模量和剪切模量通常一起预测，用于计算泊松比',
        'action':    'suggest',
    },
]


def build_graph(mae_matrix: dict) -> nx.DiGraph:
    G = nx.DiGraph()

    # ── 添加节点 ──────────────────────────────────────────
    for name, attrs in MODELS.items():
        G.add_node(f'Model:{name}', **attrs, label=name)

    for name, attrs in PROPERTIES.items():
        G.add_node(f'Property:{name}', **attrs, label=name)

    for name, attrs in CRYSTAL_SYSTEMS.items():
        G.add_node(f'CrystalSystem:{name}', **attrs, label=name)

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
        parts = key.rsplit('_', 1)
        if len(parts) != 2:
            # 处理带下划线的性质名如 formation_energy
            # key格式: {晶系}_{性质}，性质可能含下划线
            # 尝试匹配晶系名
            cs = None
            prop = None
            for cs_name in CRYSTAL_SYSTEMS:
                if key.startswith(cs_name + '_'):
                    cs = cs_name
                    prop = key[len(cs_name) + 1:]
                    break
        else:
            cs, prop = parts

        if not cs or not prop:
            continue

        models_mae = entry.get('models', {})
        sample_count = entry.get('sample_count', 0)

        if not models_mae:
            continue

        # 最优模型
        best_model = min(models_mae, key=models_mae.get)
        best_mae   = models_mae[best_model]

        # CrystalSystem -> Model (BEST_FOR)
        if f'CrystalSystem:{cs}' in G.nodes and f'Model:{best_model}' in G.nodes:
            G.add_edge(
                f'CrystalSystem:{cs}',
                f'Model:{best_model}',
                relation   = 'BEST_FOR',
                property   = prop,
                mae        = best_mae,
                sample_count = sample_count,
            )

        # 所有模型的MAE数据也加进去
        for model_name, mae in models_mae.items():
            if f'Model:{model_name}' in G.nodes:
                G.add_edge(
                    f'CrystalSystem:{cs}',
                    f'Model:{model_name}',
                    relation     = 'HAS_COVERAGE',
                    property     = prop,
                    mae          = mae,
                    sample_count = sample_count,
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
        'MEGNet':  ['CifParser', 'spglib'],
        'M3GNet':  ['CifParser', 'spglib'],
        'ALIGNN':  ['CifParser', 'spglib'],
        'CGCNN':   ['CifParser', 'spglib'],
        'CHGNet':  ['CifParser', 'spglib'],
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


def save_graph(G: nx.DiGraph, pkl_path: str, json_path: str):
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
        'edges': [
            {'source': u, 'target': v, **{k: v2 for k, v2 in data.items() if isinstance(v2, (str, int, float, bool))}}
            for u, v, data in G.edges(data=True)
        ],
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(graph_json, f, indent=2, ensure_ascii=False)
    print(f'可读版本已保存至 {json_path}')


def main():
    print('加载 MAE 矩阵...')
    with open(MAE_MATRIX_PATH) as f:
        mae_matrix = json.load(f)
    print(f'共 {len(mae_matrix)} 条路由记录')

    print('构建知识图谱...')
    G = build_graph(mae_matrix)

    print_graph_summary(G)
    save_graph(G, KG_OUTPUT_PATH, KG_JSON_PATH)


if __name__ == '__main__':
    main()