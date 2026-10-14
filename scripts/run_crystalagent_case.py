"""Run one reproducible CrystalAgent manuscript case and save its full trace."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(os.environ.get('CRYSTALAGENT_ROOT', Path(__file__).resolve().parents[1])).resolve()
BACKEND = ROOT / "backend"
OUTPUT_DIR = ROOT / "data" / "case_studies"

CASES = {
    "route_switch_lifepo4": {
        "formula": "LiFePO4",
        "mp_id": "mp-1176761",
        "purpose": (
            "Complex orthorhombic material demonstrating property-specific routing "
            "to different frozen models"
        ),
        "query": (
            "Predict the formation energy and band gap of orthorhombic LiFePO4 using "
            "the provided crystal structure. Explain why the two target properties "
            "are routed to different frozen models, and report the routing-construction "
            "MAEs, sample counts, and all physical-validation warnings."
        ),
    },
    "edge_ceo2": {
        "formula": "CeO2",
        "mp_id": "mp-20194",
        "purpose": "Extrapolation-risk detection for a rare-earth-containing structure",
        "query": (
            "Predict the formation energy and band gap of cubic CeO2 using the provided "
            "crystal structure. Report the routed models and all extrapolation or "
            "physical-validation warnings."
        ),
    },
    "edge_negative_gap_prussian_blue": {
        "formula": "Na2FeCu(CN)6",
        "mp_id": "mp-21259",
        "purpose": (
            "Complex Prussian-blue-type coordination framework used to audit "
            "negative band-gap correction and metallic interpretation"
        ),
        "query": (
            "Predict the formation energy and band gap of cubic Na2FeCu(CN)6 using "
            "the provided crystal structure. Preserve the raw band-gap output, apply "
            "the configured physical-range validation, and report any correction, "
            "warning, routed model, routing-construction MAE, and sample count."
        ),
    },
    "external_mosi2n4": {
        "formula": "MoSi2N4",
        "mp_id": None,
        "structure_path": str(
            ROOT / "data/case_studies/structures/MoSi2N4-AA.cif"
        ),
        "literature_doi": "10.1126/science.abb7023",
        "structure_source_doi": "10.1002/pssr.202200007",
        "purpose": (
            "External literature structure absent from the project's "
            "Materials Project-derived formula index"
        ),
        "query": (
            "Predict the formation energy and band gap of the literature-reported "
            "MoSi2N4 structure supplied as a CIF. Report the routed models and all "
            "physical-validation warnings, and state that the structure is external "
            "to the Materials Project-derived index."
        ),
    },
}


def json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=sorted(CASES))
    args = parser.parse_args()
    case = CASES[args.case]

    # Use the project-configured credential without copying it into this script,
    # command output, or result JSON.
    if not os.environ.get("DEEPSEEK_API_KEY"):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        match = re.search(r"DEEPSEEK_API_KEY\s*[：:]\s*([^\s]+)", readme)
        if match:
            os.environ["DEEPSEEK_API_KEY"] = match.group(1)

    sys.path.insert(0, str(BACKEND))
    from graph_orchestrator import GraphOrchestrator

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / f"{args.case}.json"
    record = {
        "case_id": args.case,
        "formula": case["formula"],
        "mp_id": case["mp_id"],
        "purpose": case["purpose"],
        "query": case["query"],
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
    }
    output.write_text(json.dumps(record, indent=2, ensure_ascii=False))

    try:
        orchestrator = GraphOrchestrator()
        if case.get("structure_path"):
            structure_file = Path(case["structure_path"]).read_bytes()
            result = orchestrator.run(
                user_input=case["query"],
                structure_file=structure_file,
            )
            record["structure_path"] = case["structure_path"]
            record["literature_doi"] = case.get("literature_doi")
            record["structure_source_doi"] = case.get("structure_source_doi")
        else:
            result = orchestrator.run(
                user_input=case["query"],
                selected_mp_id=case["mp_id"],
            )
        record["status"] = "completed" if not result.get("error") else "failed"
        record["result"] = result
    except Exception as exc:
        record["status"] = "failed"
        record["exception"] = str(exc)
        record["traceback"] = traceback.format_exc()
    finally:
        record["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        output.write_text(
            json.dumps(record, indent=2, ensure_ascii=False, default=json_default)
        )
        print(f"CASE_STATUS={record['status']}")
        print(f"CASE_OUTPUT={output}")


if __name__ == "__main__":
    main()
