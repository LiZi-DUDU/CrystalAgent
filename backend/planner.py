"""
基于知识图谱的 Planner
输入：晶系 + 目标性质列表
输出：有序工具链（包含模型选择、前处理工具、执行顺序）
"""
import pickle
import networkx as nx
from typing import Optional


class KGPlanner:
    def __init__(self, kg_path: str):
        with open(kg_path, 'rb') as f:
            self.G: nx.DiGraph = pickle.load(f)
        print(f'知识图谱加载完成：{self.G.number_of_nodes()} 节点，{self.G.number_of_edges()} 边')

    # ── 查询接口 ──────────────────────────────────────────

    def get_best_model(self, crystal_system: str, prop: str) -> Optional[dict]:
        """查询某晶系某性质的最优模型"""
        cs_node = f'CrystalSystem:{crystal_system}'
        best = None
        best_mae = float('inf')

        for u, v, data in self.G.edges(data=True):
            if (u == cs_node
                    and data.get('relation') == 'BEST_FOR'
                    and data.get('property') == prop):
                if data['mae'] < best_mae:
                    best_mae  = data['mae']
                    best = {
                        'model':        v.replace('Model:', ''),
                        'mae':          data['mae'],
                        'sample_count': data.get('sample_count', 0),
                    }
        return best

    def get_all_models_for(self, crystal_system: str, prop: str) -> list:
        """查询某晶系某性质下所有模型的MAE，按MAE排序"""
        cs_node = f'CrystalSystem:{crystal_system}'
        results = []

        for u, v, data in self.G.edges(data=True):
            if (u == cs_node
                    and data.get('relation') == 'HAS_COVERAGE'
                    and data.get('property') == prop):
                model_name = v.replace('Model:', '')
                model_data = self.G.nodes.get(f'Model:{model_name}', {})
                if model_data.get('is_available', False):
                    results.append({
                        'model':        model_name,
                        'mae':          data['mae'],
                        'sample_count': data.get('sample_count', 0),
                    })

        return sorted(results, key=lambda x: x['mae'])

    def get_physical_implications(self, prop: str) -> list:
        """查询某性质预测后的物理含义关系"""
        prop_node = f'Property:{prop}'
        implications = []

        for u, v, data in self.G.edges(data=True):
            if u == prop_node and data.get('relation') == 'PHYSICALLY_IMPLIES':
                implications.append({
                    'from':       prop,
                    'to':         v.replace('Property:', ''),
                    'condition':  data.get('condition'),
                    'implication': data.get('implication'),
                    'action':     data.get('action'),
                })
        return implications

    def get_property_info(self, prop: str) -> dict:
        """获取性质的物理约束信息"""
        return self.G.nodes.get(f'Property:{prop}', {})

    def get_model_info(self, model_name: str) -> dict:
        """获取模型信息"""
        return self.G.nodes.get(f'Model:{model_name}', {})

    # ── 核心规划方法 ──────────────────────────────────────

    def plan(self, crystal_system: str, target_properties: list) -> dict:
        """
        生成完整工具链
        返回:
        {
            'steps': [执行步骤列表],
            'warnings': [规划警告],
            'model_assignments': {prop: {model, mae, sample_count}},
        }
        """
        steps    = []
        warnings = []
        model_assignments = {}

        # ── 步骤1：前处理工具 ────────────────────────────
        steps.append({
            'step':        1,
            'type':        'tool',
            'tool':        'CifParser',
            'description': '解析结构文件，获取 pymatgen Structure 对象',
            'input':       'structure_file',
            'output':      'structure',
        })
        steps.append({
            'step':        2,
            'type':        'tool',
            'tool':        'spglib',
            'description': f'确认晶系为 {crystal_system}，获取空间群信息',
            'input':       'structure',
            'output':      'crystal_system, space_group',
        })

        # ── 步骤2：对每个目标性质规划模型 ───────────────
        step_idx = 3
        for prop in target_properties:
            best = self.get_best_model(crystal_system, prop)
            all_models = self.get_all_models_for(crystal_system, prop)
            prop_info  = self.get_property_info(prop)

            if not best:
                warnings.append(f'{crystal_system}/{prop} 在知识图谱中无对应模型，将使用集成模式')
                steps.append({
                    'step':        step_idx,
                    'type':        'ensemble',
                    'property':    prop,
                    'models':      [m['model'] for m in all_models],
                    'description': f'集成模式预测 {prop}（无最优模型记录）',
                    'input':       'structure',
                    'output':      f'predicted_{prop}',
                })
            else:
                model_assignments[prop] = best
                model_info = self.get_model_info(best['model'])
                steps.append({
                    'step':         step_idx,
                    'type':         'predict',
                    'property':     prop,
                    'model':        best['model'],
                    'mae':          best['mae'],
                    'sample_count': best['sample_count'],
                    'all_models':   all_models,
                    'description':  f'使用 {best["model"]} 预测 {prop}（该晶系最优，MAE={best["mae"]}）',
                    'input':        'structure',
                    'output':       f'predicted_{prop}',
                    'model_notes':  model_info.get('notes', ''),
                })

                # 检查样本数，样本少则警告
                if best['sample_count'] < 100:
                    warnings.append(
                        f'{crystal_system}/{prop} 训练样本仅 {best["sample_count"]} 条，'
                        f'预测置信度较低，建议参考集成模式结果'
                    )

            step_idx += 1

            # ── 查询物理含义关系，决定是否追加步骤 ──────
            implications = self.get_physical_implications(prop)
            for impl in implications:
                target_prop = impl['to']
                if target_prop in target_properties:
                    steps.append({
                        'step':        step_idx,
                        'type':        'condition_check',
                        'property':    prop,
                        'condition':   impl['condition'],
                        'implication': impl['implication'],
                        'action':      impl['action'],
                        'description': f'检查 {prop} 预测结果：{impl["implication"]}',
                        'input':       f'predicted_{prop}',
                        'output':      'condition_flag',
                    })
                    step_idx += 1

        # ── 步骤3：报告生成 ──────────────────────────────
        steps.append({
            'step':        step_idx,
            'type':        'tool',
            'tool':        'LLM_reporter',
            'description': '生成三层结构化预测报告',
            'input':       [f'predicted_{p}' for p in target_properties],
            'output':      'report',
        })

        return {
            'crystal_system':    crystal_system,
            'target_properties': target_properties,
            'steps':             steps,
            'warnings':          warnings,
            'model_assignments': model_assignments,
        }

    def explain_plan(self, plan: dict):
        """打印工具链的可读解释"""
        print(f'\n===== 规划结果 =====')
        print(f'晶系: {plan["crystal_system"]}')
        print(f'目标性质: {plan["target_properties"]}')

        if plan['warnings']:
            print(f'\n⚠ 警告:')
            for w in plan['warnings']:
                print(f'  - {w}')

        print(f'\n执行步骤:')
        for step in plan['steps']:
            t = step['type']
            if t == 'tool':
                print(f'  步骤{step["step"]}: [工具] {step["tool"]} — {step["description"]}')
            elif t == 'predict':
                print(f'  步骤{step["step"]}: [预测] {step["property"]} → {step["model"]} (MAE={step["mae"]})')
                if len(step.get('all_models', [])) > 1:
                    others = [f'{m["model"]}({m["mae"]})' for m in step['all_models'][1:]]
                    print(f'           备选模型: {", ".join(others)}')
            elif t == 'ensemble':
                print(f'  步骤{step["step"]}: [集成] {step["property"]} → {step["models"]}')
            elif t == 'condition_check':
                print(f'  步骤{step["step"]}: [条件] {step["description"]}')

        print(f'\n模型分配:')
        for prop, info in plan['model_assignments'].items():
            print(f'  {prop}: {info["model"]} (MAE={info["mae"]}, n={info["sample_count"]})')


if __name__ == '__main__':
    KG_PATH = 'data/knowledge_graph/sci_tool_kg.pkl'
    planner = KGPlanner(KG_PATH)

    # 测试：六方晶系，预测形成能和带隙
    plan = planner.plan(
        crystal_system     = '六方',
        target_properties  = ['formation_energy', 'band_gap'],
    )
    planner.explain_plan(plan)

    print('\n' + '='*40)

    # 测试：三斜晶系，样本少的情况
    plan2 = planner.plan(
        crystal_system     = '三斜',
        target_properties  = ['formation_energy', 'band_gap', 'bulk_modulus'],
    )
    planner.explain_plan(plan2)