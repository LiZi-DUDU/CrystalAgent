"""
GraphOrchestrator
基于知识图谱的编排器，只用 NetworkX +  API
"""
import os
import re
import json
import pickle
import traceback
from pathlib import Path
from typing import Optional
from pymatgen.core import Structure
from pymatgen.io.cif import CifParser
import spglib
from openai import OpenAI


# ── 常量 ─────────────────────────────────────────────────
PROJECT_ROOT = Path(
    os.environ.get('CRYSTALAGENT_ROOT', Path(__file__).resolve().parents[1])
).resolve()
KG_PATH = Path(os.environ.get(
    'CRYSTALAGENT_KG_PATH',
    PROJECT_ROOT / 'data' / 'knowledge_graph' / 'sci_tool_kg.pkl',
))
FORMULA_INDEX = Path(os.environ.get(
    'CRYSTALAGENT_FORMULA_INDEX',
    PROJECT_ROOT / 'data' / 'formula_index.json',
))
STRUCTURES_DIR = Path(os.environ.get(
    'CRYSTALAGENT_STRUCTURES_DIR',
    PROJECT_ROOT / 'data' / 'structures',
))
ERROR_CACHE = Path(os.environ.get(
    'CRYSTALAGENT_ERROR_CACHE',
    PROJECT_ROOT / 'data' / 'routing_table' / 'errors_cache.json',
))

PROPERTY_UNITS = {
    'formation_energy': 'eV/atom',
    'band_gap':         'eV',
    'bulk_modulus':     'log(GPa)',
    'shear_modulus':    'log(GPa)',
}

# 物理合理范围（Safety Module 使用）
PHYSICAL_RANGES = {
    'formation_energy': (-10.0, 5.0),
    'band_gap':         (0.0,   20.0),
    'bulk_modulus':     (-1.0,  4.0),
    'shear_modulus':    (-1.0,  4.0),
}


# ══════════════════════════════════════════════════════════
#  知识图谱查询层
# ══════════════════════════════════════════════════════════
class KnowledgeGraph:
    def __init__(self, kg_path: str):
        with open(kg_path, 'rb') as f:
            self.G = pickle.load(f)
        print(f'[KG] 加载完成：{self.G.number_of_nodes()} 节点，{self.G.number_of_edges()} 边')

    @staticmethod
    def _norm_prop(p: str) -> str:
        """把 KG 边上的性质简写/全称统一为规范名（energy->formation_energy, gap->band_gap）"""
        if p in ('formation_energy', 'formation', 'energy'):
            return 'formation_energy'
        if p in ('band_gap', 'band', 'gap'):
            return 'band_gap'
        return p

    @staticmethod
    def _cs_nodes(crystal_system: str) -> list:
        """兼容旧版/新版节点命名（CrystalSystem:{cs}、{cs}_formation、{cs}_band、{cs}_band_gap）"""
        base = f'CrystalSystem:{crystal_system}'
        return [base, f'{base}_formation', f'{base}_band', f'{base}_band_gap']

    def get_best_model(self, crystal_system: str, prop: str) -> Optional[dict]:
        """查询某晶系某性质的最优模型"""
        prop = self._norm_prop(prop)
        cs_nodes = self._cs_nodes(crystal_system)
        best     = None
        best_mae = float('inf')
        for u, v, data in self.G.edges(data=True):
            if (u in cs_nodes
                    and data.get('relation') == 'BEST_FOR'
                    and self._norm_prop(data.get('property', '')) == prop):
                if data['mae'] < best_mae:
                    best_mae = data['mae']
                    best = {
                        'model':        v.replace('Model:', ''),
                        'mae':          data['mae'],
                        'sample_count': data.get('sample_count', 0),
                        'split':        data.get('split'),
                        'frozen':       data.get('frozen', False),
                        'source':       data.get('source'),
                    }
        return best

    def get_all_models(self, crystal_system: str, prop: str) -> list:
        """查询某晶系某性质下所有可用模型，按 MAE 排序"""
        prop = self._norm_prop(prop)
        cs_nodes = self._cs_nodes(crystal_system)
        results = []
        for u, v, data in self.G.edges(data=True):
            if (u in cs_nodes
                    and data.get('relation') == 'HAS_COVERAGE'
                    and self._norm_prop(data.get('property', '')) == prop):
                model_name = v.replace('Model:', '')
                model_data = self.G.nodes.get(f'Model:{model_name}', {})
                if model_data.get('is_available', False):
                    results.append({
                        'model':        model_name,
                        'mae':          data['mae'],
                        'sample_count': data.get('sample_count', 0),
                        'split':        data.get('split'),
                        'source':       data.get('source'),
                    })
        return sorted(results, key=lambda x: x['mae'])

    def get_required_tools(self, model_name: str) -> list:
        """查询模型依赖的前处理工具"""
        model_node = f'Model:{model_name}'
        tools = []
        for u, v, data in self.G.edges(data=True):
            if u == model_node and data.get('relation') == 'REQUIRES':
                tools.append(v.replace('Tool:', ''))
        return tools

    def get_physical_implications(self, prop: str) -> list:
        """查询性质预测后的物理含义关系"""
        prop_node    = f'Property:{prop}'
        implications = []
        for u, v, data in self.G.edges(data=True):
            if u == prop_node and data.get('relation') == 'PHYSICALLY_IMPLIES':
                implications.append({
                    'from':        prop,
                    'to':          v.replace('Property:', ''),
                    'condition':   data.get('condition'),
                    'implication': data.get('implication'),
                    'action':      data.get('action'),
                })
        return implications

    def get_property_info(self, prop: str) -> dict:
        return self.G.nodes.get(f'Property:{prop}', {})

    def get_model_info(self, model_name: str) -> dict:
        return self.G.nodes.get(f'Model:{model_name}', {})

    def get_crystal_system_info(self, cs: str) -> dict:
        return self.G.nodes.get(f'CrystalSystem:{cs}', {})


# ══════════════════════════════════════════════════════════
#  Safety Module
# ══════════════════════════════════════════════════════════
class SafetyModule:
    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg

    def check_input_completeness(self, structure: Structure) -> list:
        """层1：模型输入完整性"""
        warnings = []
        if len(structure) == 0:
            warnings.append('ERROR: 结构文件为空，无法预测')
        if structure.lattice is None:
            warnings.append('ERROR: 缺少晶格参数')
        elements = [str(s.specie) for s in structure]
        if len(set(elements)) == 0:
            warnings.append('ERROR: 无法识别元素信息')
        return warnings

    def check_extrapolation_risk(self, structure: Structure,
                                  crystal_system: str, prop: str) -> list:
        """层2：外推风险标记"""
        warnings = []
        cs_info      = self.kg.get_crystal_system_info(crystal_system)
        sample_count = cs_info.get('train_count', 0)
        if sample_count < 500:
            warnings.append(
                f'WARN: {crystal_system} 晶系训练样本较少（{sample_count} 条），'
                f'预测存在外推风险'
            )
        # 检查元素覆盖（简单版本：稀土和锕系元素风险较高）
        rare_earth = {'La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy',
                      'Ho','Er','Tm','Yb','Lu','Ac','Th','Pa','U','Np'}
        elements   = {str(s.specie) for s in structure}
        rare_found = elements & rare_earth
        if rare_found:
            warnings.append(
                f'WARN: 结构含稀土/锕系元素 {rare_found}，'
                f'训练数据覆盖有限，预测可信度降低'
            )
        return warnings

    def check_physical_range(self, prop: str, value: float) -> list:
        """层3：物理范围校验"""
        warnings  = []
        ranges    = PHYSICAL_RANGES.get(prop)
        if ranges is None:
            return warnings
        lo, hi = ranges
        if value < lo or value > hi:
            warnings.append(
                f'WARN: {prop} 预测值 {value:.4f} 超出物理合理范围 '
                f'[{lo}, {hi}]，结果可能不可靠'
            )
        if prop == 'band_gap' and value < 0:
            warnings.append(f'ERROR: 带隙预测值为负（{value:.4f} eV），物理上不合理')
        return warnings

    def check_feasibility(self, crystal_system: str, prop: str,
                           all_models: list) -> list:
        """层4：晶系-性质可行性"""
        warnings = []
        if not all_models:
            warnings.append(
                f'WARN: {crystal_system}/{prop} 在知识图谱中无任何模型覆盖，'
                f'将使用集成模式，不确定度较高'
            )
        elif len(all_models) == 1:
            warnings.append(
                f'INFO: {crystal_system}/{prop} 仅有一个模型（{all_models[0]["model"]}）'
                f'，无法比较，置信度受限'
            )
        return warnings

    def run_all(self, structure: Structure, crystal_system: str,
                prop: str, value: Optional[float] = None) -> list:
        """运行全部四层检查"""
        warnings = []
        warnings += self.check_input_completeness(structure)
        warnings += self.check_extrapolation_risk(structure, crystal_system, prop)
        all_models = self.kg.get_all_models(crystal_system, prop)
        warnings += self.check_feasibility(crystal_system, prop, all_models)
        if value is not None:
            warnings += self.check_physical_range(prop, value)
        return warnings


# ══════════════════════════════════════════════════════════
#  模型加载器
# ══════════════════════════════════════════════════════════
class ModelPool:
    def __init__(self):
        self._predictors = {}

    def get(self, model_name: str):
        if model_name not in self._predictors:
            self._predictors[model_name] = self._load(model_name)
        return self._predictors[model_name]

    def _load(self, model_name: str):
        import sys
        backend_dir = str(PROJECT_ROOT / 'backend')
        if backend_dir not in sys.path:
            sys.path.insert(0, backend_dir)
        if model_name == 'MEGNet':
            from models.megnet import MEGNetPredictor
            return MEGNetPredictor()
        elif model_name == 'M3GNet':
            from models.m3gnet import M3GNetPredictor
            return M3GNetPredictor()
        elif model_name == 'ALIGNN':
            from models.alignn import ALIGNNPredictor
            return ALIGNNPredictor()
        elif model_name == 'CGCNN':
            from models.cgcnn import CGCNNPredictor
            return CGCNNPredictor()
        elif model_name == 'CHGNet':
            from models.chgnet import CHGNetPredictor
            return CHGNetPredictor()
        elif model_name == 'SevenNet':
            from models.sevennet import SevenNetPredictor
            return SevenNetPredictor()
        elif model_name == 'MACE':
            from models.mace import MACEPredictor
            return MACEPredictor()
        raise ValueError(f'未知模型: {model_name}')


# ══════════════════════════════════════════════════════════
#  GraphOrchestrator 主类
# ══════════════════════════════════════════════════════════
class GraphOrchestrator:
    def __init__(self, kg_path: str = KG_PATH):
        self.kg     = KnowledgeGraph(kg_path)
        self.safety = SafetyModule(self.kg)

        if FORMULA_INDEX.exists():
            import json
            with FORMULA_INDEX.open(encoding='utf-8') as handle:
                self._formula_index = json.load(handle)
            self._formula_index_loaded = True
        else:
            self._formula_index = {}
            self._formula_index_loaded = False
        self.pool   = ModelPool()
                # 加载 per-crystal-system error stats（用于置信度计算）
        import json as _json
        try:
            with ERROR_CACHE.open(encoding='utf-8') as _f:
                self._error_cache = _json.load(_f)
        except:
            self._error_cache = {}

        self.client = OpenAI(
            api_key  = os.environ.get('DEEPSEEK_API_KEY'),
            base_url = 'https://api.deepseek.com',
        )

    # ── 步骤1：LLM 解析用户输入 ──────────────────────────
    def get_std(self, crystal_system: str, prop: str, model: str) -> float:
        """查询某模型在某晶系某性质上的误差标准差"""
        import numpy as _np
        try:
            errs = self._error_cache.get(crystal_system, {}).get(prop, {}).get(model, [])
            if errs:
                return float(_np.nanstd(_np.abs(errs)))
        except:
            pass
        return 0.0

    def parse_input(self, user_input: str) -> dict:
        """用 Anthropic API 从自然语言提取化学式和目标性质"""
        prompt = f"""从用户输入中提取信息，返回 JSON，不要其他内容：
{{
  "formula": "化学式，如 BaTiO3",
  "target_properties": ["性质列表，从 formation_energy/band_gap/bulk_modulus/shear_modulus 中选"]
}}
用户输入：{user_input}"""

        message = self.client.chat.completions.create(
            model      = 'deepseek-chat',
            max_tokens = 256,
            messages   = [{'role': 'user', 'content': prompt}],
        )
        text = message.choices[0].message.content.strip()
        # 去掉可能的 markdown 代码块
        text = text.replace('```json', '').replace('```', '').strip()
        return json.loads(text)
        if not result.get('formula'):
            result['formula'] = user_input
        return result

    # ── 步骤2：结构解析 + 晶系识别 ───────────────────────
    def parse_structure(self, structure_file: bytes) -> tuple[Structure, str, str]:
        """解析结构文件，返回 (Structure, 晶系, 空间群)"""
        content   = structure_file.decode('utf-8', errors='ignore')
        parser    = CifParser.from_str(content)
        structure = parser.parse_structures()[0]

        lattice   = structure.lattice.matrix
        positions = structure.frac_coords
        numbers   = [site.specie.Z for site in structure]
        sym_data  = spglib.get_symmetry_dataset(
            (lattice, positions, numbers), symprec=1e-3
        )

        sg = sym_data['number']
        if sg <= 2:    cs = '三斜'
        elif sg <= 15: cs = '单斜'
        elif sg <= 74: cs = '正交'
        elif sg <= 142: cs = '四方'
        elif sg <= 167: cs = '三方'
        elif sg <= 194: cs = '六方'
        else:           cs = '立方'

        space_group = f"No.{sg} ({sym_data['international']})"
        return structure, cs, space_group

    # ── 步骤3：查图谱生成执行计划 ─────────────────────────
    def kg_plan(self, crystal_system: str, target_properties: list) -> list:
        """
        纯图查询，生成有序执行步骤列表
        每个步骤包含：type/model/property/mae/tools/warnings
        """
        steps = []
        for prop in target_properties:
            best       = self.kg.get_best_model(crystal_system, prop)
            all_models = self.kg.get_all_models(crystal_system, prop)
            if best is None and all_models:
                best = all_models[0]  # Use top model sorted by MAE
            tools      = self.kg.get_required_tools(best['model']) if best else []
            implications = self.kg.get_physical_implications(prop)

            # Safety 层4：可行性检查（预测前）
            feasibility_warnings = self.safety.check_feasibility(
                crystal_system, prop, all_models
            )

            step = {
                'property':             prop,
                'model':                best['model'] if best else 'ensemble',
                'mae':                  best['mae'] if best else None,
                'sample_count':         best['sample_count'] if best else 0,
                'route_split':          best.get('split') if best else None,
                'route_frozen':         best.get('frozen', False) if best else False,
                'route_source':         best.get('source') if best else None,
                'all_models':           all_models,
                'required_tools':       tools,
                'physical_implications': implications,
                'pre_warnings':         feasibility_warnings,
            }
            steps.append(step)
        return steps

    # ── 步骤4：执行工具链 ─────────────────────────────────
    def execute(self, structure: Structure, crystal_system: str,
                plan: list) -> dict:
        """按图谱生成的计划执行预测，返回各性质的预测结果"""
        results = {}
        for step in plan:
            prop       = step['property']
            model_name = step['model']
            warnings   = list(step['pre_warnings'])

            # Safety 层1+2：输入完整性 + 外推风险
            warnings += self.safety.check_input_completeness(structure)
            warnings += self.safety.check_extrapolation_risk(
                structure, crystal_system, prop
            )

            # 执行预测
            value = None
            try:
                if model_name == 'ensemble':
                    # 集成模式：所有可用模型加权平均
                    preds   = []
                    weights = []
                    for m_info in step['all_models']:
                        try:
                            predictor = self.pool.get(m_info['model'])
                            r         = predictor.predict(structure, prop)
                            preds.append(r['value'])
                            weights.append(1.0 / (m_info['mae'] + 1e-9))
                        except Exception:
                            continue
                    if preds:
                        total  = sum(weights)
                        value  = sum(p * w for p, w in zip(preds, weights)) / total
                        import math
                        if math.isnan(value):
                            value = None
                else:
                    predictor = self.pool.get(model_name)
                    r         = predictor.predict(structure, prop)
                    import math
                    value     = r['value'] if not (isinstance(r['value'], float) and math.isnan(r['value'])) else None
            except Exception as e:
                warnings.append(f'ERROR: {prop} 预测失败 — {str(e)}')

            # 负带隙物理上不可能：先截断为 0（按金属处理），再做范围校验
            if prop == 'band_gap' and value is not None and value < 0:
                warnings.append(f'INFO: band_gap预测为负值({value:.4f} eV)，已截断为 0（按金属处理）')
                value = 0.0

            # Safety 层3：物理范围校验（预测后）
            if value is not None:
                warnings += self.safety.check_physical_range(prop, value)
            # 物理含义检查：当前预测结果是否影响后续步骤
            for impl in step.get('physical_implications', []):
                if impl['action'] == 'warn' and value is not None:
                    condition = impl['condition']
                    # 简单条件解析
                    try:
                        triggered = eval(condition.replace(prop, str(value)))
                        if triggered:
                            warnings.append(f'INFO: {impl["implication"]}')
                    except Exception:
                        pass

            if value is not None and step.get('mae') is not None:
                confidence = round(max(0, min(1, 1 - step['mae'] / (abs(value) + 3 * self.get_std(crystal_system, prop, model_name)))), 3)
            else:
                confidence = None

            results[prop] = {
                'value':        value,
                'unit':         PROPERTY_UNITS.get(prop, ''),
                'model':        model_name,
                'mae':          step['mae'],
                'sample_count': step['sample_count'],
                'warnings':     warnings,
                'mode':         'ensemble' if model_name == 'ensemble' else 'single',
                'confidence':   confidence,
                        }

        return results

    # ── 步骤5：LLM 生成报告 ───────────────────────────────
    def generate_report(self, formula: str, crystal_system: str,
                         space_group: str, results: dict) -> str:
        """用 Anthropic API 生成三层结构化报告"""
        results_text = '\n'.join([
            f'- {prop}: {r["value"]:.4f} {r["unit"]} '
            f'(model={r["model"]}, MAE={r["mae"]}, n={r["sample_count"]})'
            f'{" [WARNINGS: " + "; ".join(r["warnings"]) + "]" if r["warnings"] else ""}'
            for prop, r in results.items() if r['value'] is not None
        ])

        prompt = f"""根据以下晶体性质预测结果生成结构化报告。

材料信息：
- 化学式：{formula}
- 晶系：{crystal_system}
- 空间群：{space_group}

预测结果：
{results_text}

请按以下三层格式生成报告（英文）：

## Layer 1: Prediction Results
列出每个性质的预测值和单位，简洁清晰。

## Layer 2: Confidence & Model Info
对每个性质说明调用的模型、该晶系历史 MAE、样本数。若为集成模式请注明。

## Layer 3: Limitations & Warnings
列出所有警告信息，说明预测的局限性和不确定性来源。"""

        if not results_text.strip():
            all_warnings = []
            for prop, r in results.items():
                all_warnings.extend(r.get('warnings', []))
            return f"[预测失败] 所有目标性质均无有效预测结果。\n警告信息:\n" + "\n".join(f"  - {w}" for w in all_warnings)

        message = self.client.chat.completions.create(
            model      = 'deepseek-chat',
            max_tokens = 1000,
            messages   = [{'role': 'user', 'content': prompt}],
        )
        return message.choices[0].message.content.strip()

    # ── 主入口 ────────────────────────────────────────────
    def find_cif_by_formula(self, formula: str, limit: int = 20) -> list:
        if not self._formula_index_loaded: return []
        mp_ids = self._formula_index.get(formula, [])
        results = []
        for mp_id in mp_ids[:limit]:
            cpath = STRUCTURES_DIR / f'{mp_id}.cif'
            if not os.path.exists(cpath): continue
            with open(cpath) as f: head = f.read(1000)
            m = re.search(r"_symmetry_Int_Tables_number\s+(\d+)", head)
            if not m: m = re.search(r"_space_group_IT_number\s+(\d+)", head)
            sg = int(m.group(1)) if m else 1
            if sg <= 2: cs = "triclinic"
            elif sg <= 15: cs = "monoclinic"
            elif sg <= 74: cs = "orthorhombic"
            elif sg <= 142: cs = "tetragonal"
            elif sg <= 167: cs = "trigonal"
            elif sg <= 194: cs = "hexagonal"
            else: cs = "cubic"
            results.append({"mp_id": mp_id, "formula": formula, "crystal_system": cs, "space_group_number": sg})
        return results

    def run(self, user_input: str, structure_file: bytes = None, selected_mp_id: str = None) -> dict:
        """
        完整编排流程
        输入：用户自然语言 + 结构文件字节
        输出：{report, results, plan, crystal_system, space_group}
        """
        print('\n[Orchestrator] 开始编排...')

        # 1. LLM 解析输入
        print('[1] 解析用户输入...')
        parsed = self.parse_input(user_input)
        formula           = parsed.get('formula', 'Unknown')
        if not formula or formula == 'Unknown':
            formula = None
        target_properties = parsed.get('target_properties', [])
        print(f'    化学式: {formula}，目标性质: {target_properties}')

        # 2. 结构解析 + 晶系识别
        print('[2] 解析结构文件...')
        if structure_file is None and selected_mp_id:
            cpath = STRUCTURES_DIR / f'{selected_mp_id}.cif'
            if not os.path.exists(cpath):
                return {"error": "CIF not found", "report": None}
            with open(cpath, "rb") as f: structure_file = f.read()
        structure, crystal_system, space_group = self.parse_structure(structure_file)
        if not formula:
            formula = structure.composition.reduced_formula
        print(f'    化学式: {formula}，晶系: {crystal_system}，空间群: {space_group}')

        # Safety 层1：输入完整性
        input_warnings = self.safety.check_input_completeness(structure)
        if any(w.startswith('ERROR') for w in input_warnings):
            return {'error': input_warnings, 'report': None}

        # 3. 查图谱生成执行计划
        print('[3] 查询知识图谱生成执行计划...')
        plan = self.kg_plan(crystal_system, target_properties)
        for step in plan:
            print(f'    {step["property"]} → {step["model"]} (MAE={step["mae"]})')
            if step.get("required_tools"):
                print(f'      工具链: {step["required_tools"]}')


        # 4. 执行预测
        print('[4] 执行预测...')
        results = self.execute(structure, crystal_system, plan)
        for prop, r in results.items():
            print(f'    {prop}: {r["value"]} {r["unit"]}')
            if r['warnings']:
                for w in r['warnings']:
                    print(f'    ⚠ {w}')
            # 根据实际预测值过滤物理含义
            for step_i in plan:
                if step_i["property"] == prop:
                    for p in step_i.get("physical_implications", []):
                        cond = p.get("condition", "")
                        if cond == "formation_energy > 0" and r["value"] > 0:
                            print(f'      [{p.get("action","")}] {p.get("implication","")[:80]}')
                        elif cond == "formation_energy < 0" and r["value"] < 0:
                            print(f'      [{p.get("action","")}] {p.get("implication","")[:80]}')
                        elif cond == "band_gap > 0" and r["value"] > 0:
                            print(f'      [{p.get("action","")}] {p.get("implication","")[:80]}')
                        elif cond == "band_gap == 0" and r["value"] == 0:
                            print(f'      [{p.get("action","")}] {p.get("implication","")[:80]}')

        # 5. 生成报告
        print('[5] 生成报告...')
        report = self.generate_report(formula, crystal_system, space_group, results)

        return {
            'formula':        formula,
            'crystal_system': crystal_system,
            'space_group':    space_group,
            'plan':           plan,
            'results':        results,
            'report':         report,
        }


# ══════════════════════════════════════════════════════════
#  测试入口
# ══════════════════════════════════════════════════════════
if __name__ == '__main__':
    import sys
    backend_dir = str(PROJECT_ROOT / 'backend')
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)

    orchestrator = GraphOrchestrator()

    # 用一个已有的 CIF 文件测试
    test_cif = STRUCTURES_DIR / 'mp-22862.cif'
    with open(test_cif, 'rb') as f:
        cif_bytes = f.read()

    result = orchestrator.run(
        user_input     = 'Predict the formation energy and band gap of NaCl',
        structure_file = cif_bytes,
    )

    print('\n' + '='*60)
    print('REPORT')
    print('='*60)
    print(result['report'])
