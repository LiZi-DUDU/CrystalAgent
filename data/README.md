# Runtime artifacts

This directory contains only the small frozen artifacts needed to execute the
reported router:

- `routing_table/mae_matrix.json`: routing-construction MAEs and frozen routes;
- `routing_table/result.json`: merged descriptive and strategy statistics;
- `knowledge_graph/sci_tool_kg.json` and `.pkl`: auditable graph and runtime graph;
- `reference/element_reference_energies.json`: shared elemental references used
  to convert SevenNet and MACE total energies to formation energies;
- `chgnet_finetune/model_108056.pt`: frozen supervised band-gap head.

Selected per-structure predictions, bootstrap source files, LLM4Mat-Bench
end-to-end results and case-study traces used by the manuscript are committed.
Their contents and scope are documented in `PAPER_RESULTS.md`. Rendered figures,
plotting scripts and manuscripts remain excluded.

The default branch does not contain the full Materials Project CIF library or
the two split manifests. Put CIF files in `data/structures/`; the download script
expects `train_meta.json` and `eval_meta.json` in `data/routing_table/`. The
historical complete snapshot is available on the public branch
`archive/pre-clean-main-20261014`.
