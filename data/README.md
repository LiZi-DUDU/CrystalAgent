# Runtime artifacts

This directory contains only the small frozen artifacts needed to execute the
reported router:

- `routing_table/mae_matrix.json`: routing-construction MAEs and frozen routes;
- `routing_table/result.json`: merged descriptive and strategy statistics;
- `knowledge_graph/sci_tool_kg.json` and `.pkl`: auditable graph and runtime graph;
- `reference/element_reference_energies.json`: shared elemental references used
  to convert SevenNet and MACE total energies to formation energies;
- `chgnet_finetune/model_108056.pt`: frozen supervised band-gap head.

Materials Project CIF files, split metadata, prediction caches, benchmark
outputs and figures are deliberately not committed. Put CIF files in
`data/structures/`. The download script expects `train_meta.json` and
`eval_meta.json` in `data/routing_table/`; these manifests can be supplied by
the user under the Materials Project data terms.
