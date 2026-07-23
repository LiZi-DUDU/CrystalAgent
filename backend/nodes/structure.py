"""
结构解析节点：用spglib从CIF/POSCAR文件中确定晶系和空间群
"""
import spglib
from pymatgen.core import Structure
from pymatgen.io.cif import CifParser
from pymatgen.io.vasp import Poscar
import io


CRYSTAL_SYSTEM_MAP = {
    "triclinic":    "三斜",
    "monoclinic":   "单斜",
    "orthorhombic": "正交",
    "tetragonal":   "四方",
    "trigonal":     "三方",
    "hexagonal":    "六方",
    "cubic":        "立方",
}


def parse_structure(state: dict) -> dict:
    file_bytes = state.get("structure_file")
    if not file_bytes:
        return {"error": "未收到结构文件"}

    try:
        # 尝试解析为CIF或POSCAR
        structure = _load_structure(file_bytes)

        # 转换为spglib格式
        lattice = structure.lattice.matrix
        positions = structure.frac_coords
        numbers = [site.specie.Z for site in structure]
        cell = (lattice, positions, numbers)

        # 获取晶系和空间群
        sym_data = spglib.get_symmetry_dataset(cell, symprec=1e-3)
        if sym_data is None:
            return {"error": "spglib 无法识别晶体对称性，请检查结构文件"}

        spacegroup_number = sym_data["number"]
        crystal_system = _spacegroup_to_crystal_system(spacegroup_number)

        return {
            "crystal_system": crystal_system,
            "space_group": f"No.{spacegroup_number} ({sym_data['international']})",
        }

    except Exception as e:
        return {"error": f"结构文件解析失败：{str(e)}"}


def _load_structure(file_bytes: bytes) -> Structure:
    """尝试以CIF或POSCAR格式解析结构文件"""
    content = file_bytes.decode("utf-8", errors="ignore")
    if "_cell_length_a" in content or "data_" in content:
        parser = CifParser.from_str(content)
        return parser.parse_structures()[0]
    else:
        poscar = Poscar.from_str(content)
        return poscar.structure


def _spacegroup_to_crystal_system(sg_number: int) -> str:
    if sg_number <= 2:   return "三斜"
    if sg_number <= 15:  return "单斜"
    if sg_number <= 74:  return "正交"
    if sg_number <= 142: return "四方"
    if sg_number <= 167: return "三方"
    if sg_number <= 194: return "六方"
    return "立方"
