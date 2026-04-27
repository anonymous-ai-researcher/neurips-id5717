#!/usr/bin/env python3
"""Generate TNN robustness benchmark instances.

For each (network, epsilon) pair:
  1. Run CMS-TNN on all correctly classified MNIST test images.
  2. Sample 4 UNSAT + 4 SAT instances uniformly at random.
  3. Encode each instance as a CNF-XOR-TNN formula.
"""

import argparse
import json
import os
import random
import struct
import subprocess
import sys
from pathlib import Path

import torch
from torchvision import datasets, transforms


RADII = [1, 2, 3]
INSTANCES_PER_POOL = 4  # 4 UNSAT + 4 SAT = 8 per (network, epsilon)
SELECTION_SEED = 42
PRESOLVER_TIMEOUT = 500  # seconds


def load_network(net_dir):
    """Load ternary weights and metadata from a trained network directory."""
    meta_path = net_dir / "metadata.json"
    with open(meta_path) as f:
        metadata = json.load(f)

    weights, biases = [], []
    for i in range(len(metadata["layer_sizes"]) - 1):
        w = torch.load(net_dir / f"layer_{i}_weights.pt", weights_only=True)
        b = torch.load(net_dir / f"layer_{i}_bias.pt", weights_only=True)
        weights.append(w.int().tolist())
        biases.append(b.int().tolist())

    return metadata, weights, biases


def encode_robustness_query(weights, biases, input_image, true_label,
                             epsilon, output_path):
    """Encode a TNN robustness query as a CNF-XOR-TNN formula.

    The formula is UNSAT iff the network classifies all inputs in
    B_epsilon(input_image) to the same label (i.e., the network is robust).

    Variables:
      - x_1, ..., x_784: input bits (within Hamming ball)
      - h_*: hidden neuron outputs
      - y_*: output neuron scores

    The formula encodes:
      1. Input constraint: Hamming(x, x*) <= epsilon
      2. TNN computation: CTCCs for each neuron
      3. Adversarial objective: exists j != true_label such that
         score(j) >= score(true_label)
    """
    n_inputs = len(input_image)
    var_counter = [0]

    def new_var():
        var_counter[0] += 1
        return var_counter[0]

    clauses = []      # CNF clauses
    ctccs = []        # CTCC constraints
    xor_clauses = []  # XOR constraints (for counting only)

    # --- Input variables ---
    input_vars = [new_var() for _ in range(n_inputs)]

    # --- Hamming ball constraint: sum |x_i - x*_i| <= epsilon ---
    # Encode using a cardinality constraint on the flipped bits
    flip_vars = []
    for i in range(n_inputs):
        f = new_var()
        flip_vars.append(f)
        # f_i <-> (x_i XOR x*_i)
        if input_image[i] == 1:
            # f_i <-> NOT x_i: (f_i OR x_i) AND (NOT f_i OR NOT x_i)
            clauses.append([f, input_vars[i]])
            clauses.append([-f, -input_vars[i]])
        else:
            # f_i <-> x_i: (f_i OR NOT x_i) AND (NOT f_i OR x_i)
            clauses.append([f, -input_vars[i]])
            clauses.append([-f, input_vars[i]])

    # sum(flip_vars) <= epsilon via CTCC
    # This is a conditional cardinality: ly <-> sum >= (epsilon+1)
    # Then add clause (NOT ly) to enforce sum < epsilon+1, i.e., sum <= epsilon
    card_output = new_var()
    ctccs.append({
        "L_plus": flip_vars,
        "L_minus": [],
        "l_y": card_output,
        "k": epsilon + 1,
    })
    clauses.append([-card_output])  # Enforce NOT ly -> sum <= epsilon

    # --- Hidden layers: CTCC per neuron ---
    prev_vars = input_vars
    for layer_idx in range(len(weights) - 1):  # All but output layer
        W = weights[layer_idx]
        b = biases[layer_idx]
        curr_vars = []
        for neuron_idx in range(len(W)):
            out_var = new_var()
            curr_vars.append(out_var)

            L_plus = [prev_vars[j] for j in range(len(prev_vars))
                       if W[neuron_idx][j] == 1]
            L_minus = [prev_vars[j] for j in range(len(prev_vars))
                        if W[neuron_idx][j] == -1]
            # Threshold k: output is 1 iff sum(L+) - sum(L-) >= k
            # With bias b, threshold is adjusted
            k = -b[neuron_idx]  # Adjust for bias

            ctccs.append({
                "L_plus": L_plus,
                "L_minus": L_minus,
                "l_y": out_var,
                "k": k,
            })
        prev_vars = curr_vars

    # --- Output layer: compute scores ---
    W_out = weights[-1]
    b_out = biases[-1]
    n_classes = len(W_out)
    score_vars = {}
    for c in range(n_classes):
        score_var = new_var()
        score_vars[c] = score_var
        L_plus = [prev_vars[j] for j in range(len(prev_vars))
                   if W_out[c][j] == 1]
        L_minus = [prev_vars[j] for j in range(len(prev_vars))
                    if W_out[c][j] == -1]
        k = -b_out[c]
        ctccs.append({
            "L_plus": L_plus,
            "L_minus": L_minus,
            "l_y": score_var,
            "k": k,
        })

    # --- Adversarial objective ---
    # Exists j != true_label: score(j) >= score(true_label)
    # Encoded as: OR over j != true_label of (score_j AND NOT score_true)
    # Simplified: at least one adversarial class has higher score
    adv_vars = []
    for j in range(n_classes):
        if j == true_label:
            continue
        a = new_var()
        adv_vars.append(a)
        # a_j -> score_j
        clauses.append([-a, score_vars[j]])
        # a_j -> NOT score_true_label
        clauses.append([-a, -score_vars[true_label]])

    # At least one adversarial class succeeds
    clauses.append(adv_vars)

    # --- Write CNF-TNN file ---
    n_vars = var_counter[0]
    with open(output_path, "w") as f:
        f.write(f"p cnf-tnn {n_vars} {len(clauses)} {len(ctccs)}\n")
        for clause in clauses:
            f.write(" ".join(str(l) for l in clause) + " 0\n")
        for ctcc in ctccs:
            lp = " ".join(str(l) for l in ctcc["L_plus"])
            lm = " ".join(str(l) for l in ctcc["L_minus"])
            f.write(f"ct {ctcc['l_y']} {ctcc['k']} "
                    f"{len(ctcc['L_plus'])} {lp} "
                    f"{len(ctcc['L_minus'])} {lm}\n")

    return n_vars, len(clauses), len(ctccs)


def main():
    parser = argparse.ArgumentParser(
        description="Generate TNN robustness benchmark instances")
    parser.add_argument("--networks", type=str, required=True,
                        help="Directory containing trained networks")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for benchmark instances")
    parser.add_argument("--instances-per-pair", type=int, default=8,
                        help="Instances per (network, epsilon) pair")
    parser.add_argument("--radii", type=int, nargs="+", default=RADII,
                        help="Perturbation radii")
    parser.add_argument("--seed", type=int, default=SELECTION_SEED,
                        help="Random seed for instance selection")
    parser.add_argument("--solver", type=str,
                        default="src/solver/build/cms-tnn",
                        help="Path to CMS-TNN binary")
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output, exist_ok=True)

    # Load MNIST test set
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x > 0.5).float()),
    ])
    test_set = datasets.MNIST("./data", train=False, transform=transform)

    networks_dir = Path(args.networks)
    net_dirs = sorted([d for d in networks_dir.iterdir() if d.is_dir()
                       and (d / "metadata.json").exists()])

    total_instances = 0
    manifest = []

    for net_dir in net_dirs:
        metadata, weights, biases = load_network(net_dir)
        net_name = net_dir.name
        print(f"\nProcessing {net_name} "
              f"(acc={metadata['test_accuracy']:.1f}%)")

        for epsilon in args.radii:
            instance_dir = Path(args.output) / net_name / f"eps{epsilon}"
            os.makedirs(instance_dir, exist_ok=True)

            # Pre-solve to classify UNSAT/SAT
            sat_pool, unsat_pool = [], []
            for idx in range(len(test_set)):
                img, label = test_set[idx]
                img_binary = (img.flatten() > 0.5).int().tolist()

                tmp_path = instance_dir / f"_tmp_{idx}.cnf-tnn"
                encode_robustness_query(
                    weights, biases, img_binary, label,
                    epsilon, str(tmp_path))

                # Quick solve to classify
                try:
                    result = subprocess.run(
                        [args.solver, "--no-proof", str(tmp_path)],
                        capture_output=True, text=True,
                        timeout=PRESOLVER_TIMEOUT)
                    if "s UNSATISFIABLE" in result.stdout:
                        unsat_pool.append(idx)
                    elif "s SATISFIABLE" in result.stdout:
                        sat_pool.append(idx)
                except subprocess.TimeoutExpired:
                    pass
                finally:
                    tmp_path.unlink(missing_ok=True)

                if len(sat_pool) >= 100 and len(unsat_pool) >= 100:
                    break  # Enough candidates

            # Sample instances
            n_per_pool = args.instances_per_pair // 2
            selected_unsat = random.sample(
                unsat_pool, min(n_per_pool, len(unsat_pool)))
            selected_sat = random.sample(
                sat_pool, min(n_per_pool, len(sat_pool)))

            for pool_name, indices in [("unsat", selected_unsat),
                                        ("sat", selected_sat)]:
                for i, idx in enumerate(indices):
                    img, label = test_set[idx]
                    img_binary = (img.flatten() > 0.5).int().tolist()

                    out_path = (instance_dir /
                                f"{pool_name}_{i:02d}_img{idx}.cnf-tnn")
                    n_vars, n_cls, n_ctcc = encode_robustness_query(
                        weights, biases, img_binary, label,
                        epsilon, str(out_path))

                    manifest.append({
                        "network": net_name,
                        "epsilon": epsilon,
                        "type": pool_name,
                        "mnist_index": idx,
                        "true_label": label,
                        "path": str(out_path),
                        "n_vars": n_vars,
                        "n_clauses": n_cls,
                        "n_ctccs": n_ctcc,
                    })
                    total_instances += 1

            print(f"  eps={epsilon}: {len(selected_unsat)} UNSAT + "
                  f"{len(selected_sat)} SAT instances")

    # Save manifest
    with open(Path(args.output) / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nGenerated {total_instances} benchmark instances.")


if __name__ == "__main__":
    main()
