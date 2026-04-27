# Certified Reasoning for Ternary Neural Networks

Artifact for reproducing the experiments in *Efficient Certified Reasoning for Ternary Neural Networks*.

## Overview

This artifact provides:

1. **CryptoMiniSat-TNN (CMS-TNN)**: A CDCL SAT solver extended with native Conditional Ternary Cardinality Constraint (CTCC) propagation.
2. **FRAT-xor-tnn**: A proof elaborator that converts FRAT-XOR-TNN proofs to the XLRUP format.
3. **cake_xlrup-TNN**: A CakeML-verified proof checker for XLRUP proofs with TNN-specific `clause-from-tnn` steps.
4. **ApproxMCCert-TNN**: A certified approximate model counter for CNF-TNN formulas.
5. **CertCheck-TNN**: A certificate checker for counting results with PAC guarantees.
6. **Benchmark suite**: 1,080 TNN robustness queries across 45 networks (3 architectures x 3 sparsity levels x 5 seeds).
7. **FashionMNIST generalization**: Cross-dataset evaluation on FashionMNIST (360 instances, Small architecture).
8. **Isabelle/HOL proofs**: Locale instantiation for the PAC guarantee (Theorem 4.1).

## Directory Structure

```
.
├── README.md
├── configs/
│   ├── experiment.yaml        # Main experiment configuration
│   └── pac_parameters.yaml    # PAC parameter settings
├── src/
│   ├── solver/                # CryptoMiniSat-TNN source
│   │   ├── CMakeLists.txt
│   │   └── ctcc_propagator.cpp
│   ├── elaborator/            # FRAT-xor-tnn elaborator
│   │   └── frat_xor_tnn.cpp
│   ├── checker/               # cake_xlrup-TNN checker
│   │   └── cake_xlrup_tnn.cml
│   ├── counter/               # ApproxMCCert-TNN counter
│   │   └── approxmc_tnn.cpp
│   └── certcheck/             # CertCheck-TNN verifier
│       └── certcheck_tnn.cpp
├── scripts/
│   ├── train_tnn.py           # TTQ training script
│   ├── generate_benchmark.py  # Benchmark instance generation
│   ├── run_qualitative.sh     # Run RQ1 experiments
│   ├── run_quantitative.sh    # Run RQ2 experiments
│   ├── run_ablation.sh        # Run RQ3 experiments
│   ├── run_all.sh             # Run all experiments
│   ├── collect_results.py     # Parse logs and collect results
│   └── generate_tables.py     # Generate LaTeX tables and figures
├── benchmarks/                # Generated benchmark instances
├── isabelle/                  # Isabelle/HOL formalization
│   └── CTCC_Locale.thy
└── requirements.txt
```

## Requirements

### Hardware
- CPU: AMD EPYC-Milan (or equivalent x86-64) recommended
- RAM: 512 GB (16 GB per instance)
- Storage: ~50 GB for full experiment output

### Software
- Linux (Ubuntu 22.04+ recommended)
- GCC 12+ or Clang 15+
- CMake 3.20+
- Python 3.11+
- PyTorch 2.2+
- CakeML compiler (for checker build)
- Isabelle/HOL 2024 (for proof verification only)

### Python Dependencies

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Build all tools

```bash
# Build CryptoMiniSat-TNN
cd src/solver && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
cd ../../..

# Build FRAT-xor-tnn elaborator
cd src/elaborator && make
cd ../..

# Build cake_xlrup-TNN (requires CakeML compiler)
cd src/checker && make
cd ../..

# Build ApproxMCCert-TNN
cd src/counter && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
cd ../../..

# Build CertCheck-TNN
cd src/certcheck && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
cd ../../..
```

### 2. Train TNN networks and generate benchmark

```bash
# Train 45 TNN networks (3 architectures x 3 sparsity levels x 5 seeds)
python scripts/train_tnn.py --output benchmarks/networks/

# Generate 1,080 benchmark instances
python scripts/generate_benchmark.py \
    --networks benchmarks/networks/ \
    --output benchmarks/instances/ \
    --instances-per-pair 8 \
    --radii 1 2 3 \
    --seed 42
```

### 3. Run experiments

```bash
# Run all experiments (estimated wall-clock: ~18h on 128-thread EPYC-Milan)
bash scripts/run_all.sh

# Or run individual research questions:
bash scripts/run_qualitative.sh    # RQ1: ~12h
bash scripts/run_quantitative.sh   # RQ2: ~24h
bash scripts/run_ablation.sh       # RQ3: ~6h
```

### 4. Collect results and generate tables

```bash
python scripts/collect_results.py --results results/ --output results/summary.json
python scripts/generate_tables.py --summary results/summary.json --output tables/
```

## Experiment Details

### Network Architectures

| Architecture | Layers | Neurons/Layer | Total Neurons |
|---|---|---|---|
| Small | 3 | 64, 64, 64 | 192 |
| Medium | 4 | 128, 256, 256, 128 | 768 |
| Large | 5 | 256, 512, 512, 512, 256 | 2,048 |

### TTQ Training Hyperparameters

| Parameter | Value |
|---|---|
| Optimizer | SGD with momentum 0.9 |
| Initial learning rate | 0.01 |
| LR decay | x0.1 at epochs 80, 120 |
| Batch size | 128 |
| Total epochs | 160 |
| Sparsity levels | 15%, 30%, 50% |
| Random seeds | 0, 1, 2, 3, 4 |
| Dataset | MNIST |

### PAC Parameters (Counting)

| Parameter | Symbol | Value |
|---|---|---|
| Tolerance | η | 0.8 |
| Confidence | δ | 0.2 |
| Iterations | t | 4 |
| Pivot | pivot | 23 |

### Solver Configuration

| Parameter | Qualitative | Quantitative |
|---|---|---|
| Time limit | 500 s | 5,000 s |
| Memory limit | 16 GB | 16 GB |
| Threads | 1 | 1 |

### Instance Selection

For each (network, ε) pair:
1. Run CMS-TNN (uncertified, 500s timeout) on all correctly classified MNIST test images (~9,600).
2. Sample 4 UNSAT + 4 SAT instances uniformly at random (seed 42).
3. Total: 45 networks × 8 instances × 3 radii = 1,080 instances.

## Research Questions

### RQ1: Certified Qualitative Reasoning (540 UNSAT instances)

Compares CMS-TNN + cake_xlrup-TNN against:
- CaDiCaL 2.1.0 + cake_lpr
- RoundingSat 2.0 + VeriPB 2.0

Metrics: PAR-2 (s), coverage (%), certificate size (MB), checking time (s).

### RQ2: Certified Quantitative Reasoning (1,080 instances)

Compares ApproxMCCert-TNN + CertCheck-TNN against ApproxMCCert (CNF baseline).

Metrics: PAR-2 (s), coverage (%), certificate size (MB).

### RQ3: Ablation on Native CTCC Propagation (540 UNSAT instances)

Compares CMS-TNN (native CTCC) against CMS-CNF (pre-encodes CTCCs into CNF clauses).

Metrics: PAR-2 (s), propagation count, certificate size (MB).

### RQ4: Certificate and Checker Scaling

Measures certificate size and checking time as functions of network size and perturbation radius.

### RQ5: Bug Detection

Documents a real propagation bug caught by certificate verification: three silent UNSAT misclassifications in an early solver version.

### FashionMNIST Generalization

Small-architecture TNNs (64-64-64, sparsity 15%/30%/50%, 5 seeds) trained on FashionMNIST using the same TTQ protocol. 360 instances (3 sparsities × 5 seeds × 8 instances × 3 ε).

### Z3 Baseline (SMT)

Z3 v4.13 (integer arithmetic mode) on 540 UNSAT qualitative instances as an additional uncertified baseline. Z3 does not produce checkable certificates.

## Compute Budget

- Total: ~2,400 single-core CPU-hours
- Wall-clock: ~18 hours on 128 hardware threads (AMD EPYC-Milan)
- Storage: ~50 GB for full experiment output including all certificates

## Expected Results

### RQ1 (Qualitative)
- CMS-TNN + cake: **97%** certified coverage, PAR-2 66±9 s
- 17× faster than CaDiCaL + cake_lpr, 24× faster than RoundingSat + VeriPB
- Certificates **81×** smaller (10.4 MB vs 840 MB median)

### RQ2 (Quantitative)
- ApproxMCCert-TNN: **81%** certified coverage (875/1,080)
- 4.8× lower PAR-2 than CNF baseline
- Certificates: 4.2 MB median vs 341 MB for CNF baseline

### RQ3 (Ablation)
- Native CTCC propagation: **13.4×** PAR-2 speedup over CNF pre-encoding
- 8.3× fewer propagation steps

### FashionMNIST Generalization
- CMS-TNN + cake: **100%** certified coverage, PAR-2 18±4 s
- 17× faster than CaDiCaL + cake_lpr (84% coverage)
- Certificates 271× smaller (3.1 MB vs 840 MB)

## Reproducing Specific Tables and Figures

| Paper Element | Command |
|---|---|
| Table 1 (RQ1) | `python scripts/generate_tables.py --table qual` |
| Table 2 (RQ2) | `python scripts/generate_tables.py --table quant` |
| Figure 2 (boxplot) | `python scripts/generate_tables.py --figure certsize` |
| Figure 3 (scaling) | `python scripts/generate_tables.py --figure scaling` |
| Table 3 (ablation) | `python scripts/generate_tables.py --table ablation` |
| Table 23 (FashionMNIST) | `python scripts/generate_tables.py --table fashionmnist` |

## Isabelle/HOL Verification

To verify the PAC guarantee formalization:

```bash
cd isabelle/
isabelle build -d . -b CTCC_Locale
```

This checks the locale instantiation proving that CMS-TNN and cake_xlrup-TNN satisfy the four locale conditions of the ApproxMC formalization, yielding the PAC guarantee (Theorem 4.1).

## Statistical Tests

All pairwise comparisons use the Wilcoxon signed-rank test (non-parametric, paired, two-tailed) with Bonferroni correction. Results are generated automatically by `collect_results.py`.

## License

This software is released under the MIT License. See `LICENSE` for details.
