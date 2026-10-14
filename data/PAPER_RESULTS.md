# Manuscript source results

This manifest maps the committed machine-readable results to the analyses in
the CrystalAgent manuscript. It excludes rendered figures, plotting code,
manuscript files and execution logs.

## Internal Materials Project evaluation

Files under `outputs/`:

- `r1m3_results.json`: corrected formation-energy descriptive results.
- `mae_matrix_corrected.json`: corrected per-system MAE matrix.
- `mu_corrected.json`: elemental reference energies used for universal-potential
  formation-energy conversion.
- `bg_eval_errors.json`: per-structure band-gap errors used in evaluation.
- `pred_megnet_eval.jsonl`, `pred_m3gnet_eval.jsonl`,
  `pred_alignn_eval.jsonl`, `pred_cgcnn_eval.jsonl`: direct-predictor
  formation-energy outputs on the evaluation split.
- `raw_sevennet_eval.jsonl`, `raw_mace_eval.jsonl`: universal-potential raw
  outputs on the evaluation split.
- `sg_recompute.json`: recomputed crystal-system-level evaluation data.
- `r1m4_bootstrap_results.json`: paired bootstrap comparisons and strategy
  statistics.
- `r1m4_pooled_strategy_results.json` and
  `r1m4_pooled_strategy_dist.npz`: pooled complete-case strategy comparisons and
  their bootstrap distributions.
- `r1m4_heldout_stability_results.json` and
  `r1m4_heldout_stability_dist.npz`: frozen-route agreement results and
  bootstrap distributions.
- `figure3_paired_bootstrap_source_updated.csv`: final source values for the
  paired winner-versus-runner-up analysis.
- `figure3c_formation_energy_error_correlations_source.csv`: final
  formation-energy absolute-error correlations.
- `r1m4_winner_pred_results.json`: structural-descriptor classifier and
  model-win analysis.
- `r1m4_perturbation_results.json`: lattice-scaling response analysis.
- `r1m4_audit_results.json`: auxiliary audit results.
- `../figures/update_result_sevennet_baseline.py`: statistical update script
  that recomputes the SevenNet fixed-single formation-energy comparisons. It
  does not render a figure; plotting scripts remain excluded.

The compact aggregate files in `routing_table/mae_matrix.json` and
`routing_table/result.json` are retained for runtime use and summary lookup.
The per-structure files above provide the source values required to inspect the
reported aggregate statistics.

## LLM4Mat-Bench evaluation

Files under `benchmark/` include:

- `llm4mat_published_numbers.json`: published reference values transcribed for
  the benchmark comparison.
- `llm4mat_band_gap_both_42.jsonl` and its summary: fixed seed-42 band-gap
  sample results.
- `llm4mat_formation_energy_both_42.jsonl` and its summary: fixed seed-42
  formation-energy sample results.
- `llm4mat_formation_energy_corrected_router_seed42.jsonl` and its summary:
  corrected frozen-router formation-energy results.
- `agentcmp_chemcrow_*_seed42.jsonl` and summaries: adapted ChemCrow comparison
  outputs.
- `agentcmp_chatmof_*_seed42.jsonl` and summaries: adapted ChatMOF comparison
  outputs.
- `crystalagent_e2e_kg_v2_1_seed42/manifest_*_seed42.json`: the two fixed
  1,000-structure sample manifests.
- `crystalagent_e2e_kg_v2_1_seed42/crystalagent_e2e_*_seed42.jsonl` and
  summaries: merged end-to-end CrystalAgent predictions and metrics.

ChemCrow and ChatMOF are adapted configurations rather than exact reproductions
of the original systems. Effective sample counts and failures are recorded in
the corresponding JSONL and summary files.

## Case studies

Files under `case_studies/` include the released execution traces and supplied
structures used for the manuscript examples:

- `case_study_summary.json`;
- `route_switch_lifepo4.json`;
- `external_mosi2n4.json`;
- `edge_negative_gap_prussian_blue.json`;
- `structures/MoSi2N4-AA.cif`, `MoSi2N4-AB.cif` and
  `MoSi2N4-AC.cif`.

## Split and structure snapshot

The default branch omits `train_meta.json`, `eval_meta.json` and the 154,373 CIF
files. The byte-identical historical snapshot is on the public branch
`archive/pre-clean-main-20261014`. The split contains 108,059 routing-
construction structures and 46,314 evaluation structures with no overlapping
Materials Project identifiers.

The original Materials Project database version and the random seed used to
create the 70:30 split were not recorded. File modification times must not be
interpreted as a database release identifier or a verified API access date.
