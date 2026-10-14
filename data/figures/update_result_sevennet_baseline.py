# -*- coding: utf-8 -*-
"""Make SevenNet the canonical formation-energy single-model baseline.

The script recomputes all formation-energy strategy comparisons from the
per-structure complete-case errors. It updates both the R1-M4 statistics file
and the merged routing-table result, while preserving the band-gap results and
the model-versus-model comparisons.
"""

from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path

import numpy as np


ROOT = Path(os.environ.get("CRYSTAL_ROOT", "/root/autodl-tmp/crystal_agent"))
OUTPUTS = ROOT / "data" / "outputs"
ROUTING = ROOT / "data" / "routing_table"
RESULT_PATH = ROUTING / "result.json"
R1M4_PATH = OUTPUTS / "r1m4_bootstrap_results.json"
STABILITY_PATH = OUTPUTS / "r1m4_heldout_stability_results.json"
MATRIX_PATH = ROUTING / "mae_matrix.json"

SEED = 20260815
B = 10_000
CHUNK = 400
BEST_SINGLE = "SevenNet"

CS_CN2EN = {
    "三斜": "triclinic",
    "单斜": "monoclinic",
    "正交": "orthorhombic",
    "四方": "tetragonal",
    "三方": "trigonal",
    "六方": "hexagonal",
    "立方": "cubic",
}
SYSTEMS = [
    "triclinic",
    "monoclinic",
    "orthorhombic",
    "tetragonal",
    "trigonal",
    "hexagonal",
    "cubic",
]


def load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_dump(path: Path, payload) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, ensure_ascii=False)
        handle.write("\n")
    os.replace(tmp, path)


def backup_once(path: Path) -> Path:
    backup = path.with_name(path.name + ".before_sevennet_baseline.bak")
    if not backup.exists():
        shutil.copy2(path, backup)
    return backup


def paired_boot_many(differences: list[np.ndarray]):
    """Bootstrap several paired mean differences using identical resamples."""
    arrays = [np.asarray(values, dtype=float) for values in differences]
    n = len(arrays[0])
    if n == 0 or any(len(values) != n for values in arrays):
        raise ValueError("Paired bootstrap arrays must have one shared, nonzero n")
    rng = np.random.default_rng(SEED)
    means = [np.empty(B, dtype=float) for _ in arrays]
    for start in range(0, B, CHUNK):
        count = min(CHUNK, B - start)
        indices = rng.integers(0, n, size=(count, n))
        for out, values in zip(means, arrays):
            out[start:start + count] = values[indices].mean(axis=1)

    summaries = []
    for values, boot in zip(arrays, means):
        lo, hi = np.percentile(boot, [2.5, 97.5])
        p_value = 2.0 * min(
            float((boot <= 0).mean()), float((boot >= 0).mean())
        )
        summaries.append({
            "diff": float(values.mean()),
            "ci": [float(lo), float(hi)],
            "p": min(1.0, p_value),
        })
    return summaries


def load_formation_data():
    metadata = {
        row["id"]: row
        for row in load_json(ROUTING / "eval_meta.json")
    }
    crystal_system = {
        structure_id: CS_CN2EN[row["crystal_system"]]
        for structure_id, row in metadata.items()
    }

    predictions = {}
    for key, model in {
        "megnet": "MEGNet",
        "m3gnet": "M3GNet",
        "alignn": "ALIGNN",
        "cgcnn": "CGCNN",
    }.items():
        predictions[model] = {
            row["id"]: float(row["pred"])
            for row in load_jsonl(OUTPUTS / f"pred_{key}_eval.jsonl")
            if math.isfinite(float(row["pred"]))
        }

    references = load_json(OUTPUTS / "mu_corrected.json")
    for key, model in {"sevennet": "SevenNet", "mace": "MACE"}.items():
        model_predictions = {}
        for row in load_jsonl(OUTPUTS / f"raw_{key}_eval.jsonl"):
            n_atoms = int(row["n"])
            reference_energy = sum(
                count * references.get(element, 0.0)
                for element, count in row["els"].items()
            )
            prediction = float(row["E"]) / n_atoms - reference_energy / n_atoms
            if math.isfinite(prediction):
                model_predictions[row["id"]] = prediction
        predictions[model] = model_predictions

    absolute_errors = {
        model: {
            structure_id: abs(
                prediction - float(metadata[structure_id]["formation_energy"])
            )
            for structure_id, prediction in model_predictions.items()
            if structure_id in metadata
        }
        for model, model_predictions in predictions.items()
    }

    models = ["MEGNet", "M3GNet", "ALIGNN", "CGCNN", "SevenNet", "MACE"]
    complete_ids = [
        structure_id
        for structure_id in metadata
        if all(structure_id in absolute_errors[model] for model in models)
    ]
    ids_by_system = {
        system: [
            structure_id
            for structure_id in complete_ids
            if crystal_system[structure_id] == system
        ]
        for system in SYSTEMS
    }
    return metadata, crystal_system, predictions, absolute_errors, complete_ids, ids_by_system


def frozen_configuration():
    matrix = load_json(MATRIX_PATH)
    winners = {}
    inverse_mae_weights = {}
    for key, payload in matrix.items():
        system_cn, task = key.split("_", 1)
        if task != "formation_energy":
            continue
        system = CS_CN2EN[system_cn]
        model_maes = payload["models"]
        winners[system] = min(model_maes, key=model_maes.get)
        inverse_mae_weights[system] = {
            model: 1.0 / max(float(mae), 1e-6)
            for model, mae in model_maes.items()
        }
    if set(winners) != set(SYSTEMS):
        raise ValueError(f"Incomplete frozen routes: {winners}")
    return winners, inverse_mae_weights


def strategy_errors(
    structure_ids,
    metadata,
    crystal_system,
    predictions,
    absolute_errors,
    winners,
    weights,
):
    router = []
    single = []
    ensemble = []
    for structure_id in structure_ids:
        system = crystal_system[structure_id]
        router.append(absolute_errors[winners[system]][structure_id])
        single.append(absolute_errors[BEST_SINGLE][structure_id])
        system_weights = weights[system]
        total_weight = sum(system_weights.values())
        ensemble_prediction = sum(
            system_weights[model] * predictions[model][structure_id]
            for model in system_weights
        ) / total_weight
        ensemble.append(abs(
            ensemble_prediction
            - float(metadata[structure_id]["formation_energy"])
        ))
    return {
        "router": np.asarray(router, dtype=float),
        "single_best": np.asarray(single, dtype=float),
        "ensemble": np.asarray(ensemble, dtype=float),
    }


def strategy_rows(strategy):
    pairs = [
        ("router", "single_best"),
        ("router", "ensemble"),
        ("ensemble", "single_best"),
    ]
    differences = [strategy[a] - strategy[b] for a, b in pairs]
    summaries = paired_boot_many(differences)
    rows = []
    for (a, b), summary in zip(pairs, summaries):
        rows.append({
            "pair": f"{a} vs {b}",
            "n": int(len(strategy[a])),
            "mae_a": float(strategy[a].mean()),
            "mae_b": float(strategy[b].mean()),
            **summary,
        })
    return rows


def main():
    result = load_json(RESULT_PATH)
    r1m4 = load_json(R1M4_PATH)
    stability = load_json(STABILITY_PATH)
    (
        metadata,
        crystal_system,
        predictions,
        absolute_errors,
        complete_ids,
        ids_by_system,
    ) = load_formation_data()
    winners, weights = frozen_configuration()

    expected_complete_n = int(result["metadata"]["formation_energy_complete_case_n"])
    if len(complete_ids) != expected_complete_n:
        raise ValueError(
            f"Complete-case mismatch: {len(complete_ids)} != {expected_complete_n}"
        )

    per_system_rows = {}
    for system in SYSTEMS:
        strategy = strategy_errors(
            ids_by_system[system], metadata, crystal_system, predictions,
            absolute_errors, winners, weights,
        )
        per_system_rows[system] = strategy_rows(strategy)

    overall_strategy = strategy_errors(
        complete_ids, metadata, crystal_system, predictions,
        absolute_errors, winners, weights,
    )
    overall_rows = strategy_rows(overall_strategy)

    all_sevennet_errors = np.asarray(
        list(absolute_errors[BEST_SINGLE].values()), dtype=float
    )
    if len(all_sevennet_errors) != int(result["metadata"]["n_eval"]):
        raise ValueError(
            f"SevenNet finite-sample mismatch: {len(all_sevennet_errors)}"
        )
    sevennet_full_mae = float(all_sevennet_errors.mean())

    r1m4.setdefault("single_best_models", {})["formation_energy"] = BEST_SINGLE
    r1m4["single_best_models"].setdefault("band_gap", "CHGNet")
    r1m4["per_system"]["formation_energy_strategies"] = per_system_rows
    r1m4["overall"]["formation_energy_strategies"] = overall_rows

    result["formation_energy_descriptive"]["overall_eval"][
        "single_best"
    ] = sevennet_full_mae
    result["formation_energy_descriptive"]["overall_eval"][
        "single_best_model"
    ] = BEST_SINGLE
    result["strategy_statistics"]["per_system"][
        "formation_energy_strategies"
    ] = per_system_rows
    result["strategy_statistics"]["overall"][
        "formation_energy_strategies"
    ] = overall_rows
    result["strategy_statistics"]["heldout_stability"] = stability
    result["metadata"]["single_best_models"] = {
        "formation_energy": BEST_SINGLE,
        "band_gap": "CHGNet",
    }
    result["metadata"]["frozen_routes"] = {
        "formation_energy": winners,
        "band_gap": stability["band_gap"]["frozen_route"],
    }
    result["sources"]["heldout_stability"] = str(
        STABILITY_PATH.relative_to(ROOT)
    )
    result["sources"]["single_best_update_script"] = (
        "data/figures/update_result_sevennet_baseline.py"
    )

    stability_fe = stability["formation_energy"]
    if stability_fe.get("single_best_model") != BEST_SINGLE:
        raise ValueError("Held-out stability file does not identify SevenNet")
    observed = next(
        row for row in overall_rows if row["pair"] == "router vs single_best"
    )
    stability_mean = stability_fe["router_gain"]["delta_single"]["mean"]
    if not math.isclose(observed["diff"], stability_mean, abs_tol=2e-5):
        raise ValueError(
            f"Pooled/stratified point estimates disagree: "
            f"{observed['diff']} vs {stability_mean}"
        )

    r1m4_backup = backup_once(R1M4_PATH)
    result_backup = backup_once(RESULT_PATH)
    atomic_dump(R1M4_PATH, r1m4)
    atomic_dump(RESULT_PATH, result)

    reduction = (
        sevennet_full_mae
        - result["formation_energy_descriptive"]["overall_eval"]["router"]
    ) / sevennet_full_mae * 100.0
    print(f"updated: {R1M4_PATH}")
    print(f"updated: {RESULT_PATH}")
    print(f"backups: {r1m4_backup}, {result_backup}")
    print(f"full finite sample: n={len(all_sevennet_errors)}")
    print(f"SevenNet MAE: {sevennet_full_mae:.12f} eV/atom")
    print(f"router reduction vs SevenNet: {reduction:.6f}%")
    print(
        "complete-case router - SevenNet: "
        f"{observed['diff']:+.12f}, "
        f"95% pooled paired-bootstrap CI "
        f"[{observed['ci'][0]:+.12f}, {observed['ci'][1]:+.12f}]"
    )
    print(
        "Figure 4d stratified-bootstrap CI: "
        f"[{stability_fe['router_gain']['delta_single']['ci'][0]:+.12f}, "
        f"{stability_fe['router_gain']['delta_single']['ci'][1]:+.12f}]"
    )


if __name__ == "__main__":
    main()
