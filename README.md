# CrystalAgent

CrystalAgent is an auditable crystal-property inference system that selects a
frozen graph-neural-network configuration from the crystal system and requested
property. The released runtime covers formation energy and band gap using
MEGNet, M3GNet, ALIGNN, CGCNN, CHGNet, SevenNet and MACE adapters.

This branch is a code-only reproducibility release. Manuscripts, plotting code,
rendered figures, logs, prediction caches and full Materials Project structures
are intentionally excluded.

## Reproducibility scope

The repository includes the source code and the small frozen artifacts required
for inference: the routing matrix, knowledge graph, elemental reference table,
CGCNN checkpoints and CHGNet band-gap head. Public model packages download their
own released checkpoints on first use. Materials Project structures and the
LLM4Mat-Bench CSV must be obtained from their original providers under the
corresponding terms.

The tested environment used Python 3.10.20, PyTorch 2.2.1, CUDA 12.4 and DGL
2.5.0+cu124. Exact package versions are listed in `requirements.txt`.

## Installation

```bash
git clone -b reproducible-release https://github.com/LiZi-DUDU/CrystalAgent.git
cd CrystalAgent
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install the DGL wheel matching your CUDA/CPU runtime first.
python -m pip install -r requirements.txt
cp .env.example .env
```

Set `DEEPSEEK_API_KEY` in the shell or load it from `.env`. The key is used only
for natural-language request parsing and report generation; numerical values are
produced by the routed material-property models.

## Run the API

```bash
export DEEPSEEK_API_KEY=your_key
./start.sh
```

The service listens on port 8004 by default. A CIF can be submitted to
`POST /predict` with `user_input` and `structure_file` form fields.

## Deterministic checks

The knowledge graph and packaged artifacts can be checked without an API key or
GPU:

```bash
python -m unittest discover -s tests -p 'test_*.py'
python scripts/build_knowledge_graph.py
```

To run model-level smoke tests after adding CIF files:

```bash
python scripts/smoke_crystalagent_core_models.py
```

## Rebuild and evaluate the router

1. Place the routing-construction and evaluation manifests at
   `data/routing_table/train_meta.json` and `eval_meta.json`.
2. Download the corresponding CIF files:

   ```bash
   python scripts/download_structures.py --api_key "$MP_API_KEY"
   ```

3. Recompute the routing table and knowledge graph:

   ```bash
   python scripts/build_routing_table.py
   python scripts/build_knowledge_graph.py
   ```

4. Evaluate the three deployment strategies:

   ```bash
   python scripts/evaluate_system.py
   ```

Generated metadata, predictions and logs are ignored by Git.

## LLM4Mat-Bench seed-42 evaluation

Clone LLM4Mat-Bench into `LLM4Mat-Bench/`, add the required CIF files to
`data/structures/`, then run:

```bash
python scripts/benchmark_llm4mat.py --n 1000 --property band_gap --mode both --seed 42
python scripts/benchmark_llm4mat.py --n 1000 --property formation_energy --mode both --seed 42
```

The instrumented end-to-end runner is
`scripts/benchmark_crystalagent_e2e_seed42_v2.py`. ChemCrow and ChatMOF were
evaluated as adapted configurations, not as exact reproductions of their
original systems; their mutually incompatible environments are documented in
`requirements-chemcrow.txt` and `requirements-chatmof.txt`.

## Security and generated files

No API keys are committed. `.env`, figures, manuscripts, plotting scripts,
large structures and generated benchmark outputs are excluded by `.gitignore`.
