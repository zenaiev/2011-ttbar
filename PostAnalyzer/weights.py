import torch
import json
import os
import sys
import argparse
sys.path.append(os.getcwd())
from train_solve_nn import SolveMLP

def create_weights(out_dir, output_path, namespace):
    model = SolveMLP()
    model.load_state_dict(torch.load(f"{out_dir}/solve_nn_best.pt", map_location="cpu"))
    model.eval()

    assert hasattr(model, 'input_proj'), "[ERROR] Немає input_proj"
    assert hasattr(model, 'blocks'),     "[ERROR] Немає blocks"
    assert hasattr(model, 'head'),       "[ERROR] Немає head"

    assert len(model.blocks) == 3, f"[ERROR] Очікується 3 ResBlocks, знайдено {len(model.blocks)}"

    with open(f"{out_dir}/norm_stats.json", "r") as f:
        norm = json.load(f)

    assert len(norm["x_mean"]) == 26, f"[ERROR] x_mean має {len(norm['x_mean'])} елементів, очікується 26"

    with open(output_path, "w") as f:
        f.write(f"#pragma once\n#include <vector>\n\nnamespace {namespace} {{\n")

        f.write("    // Нормалізація\n")
        for key in ["x_mean", "x_std"]:
            vals = norm[key]
            f.write(f"    const std::vector<float> {key} = {{ {', '.join([f'{v:.8f}f' for v in vals])} }};\n")
        f.write("\n")

        f.write("    // Ваги та LayerNorm\n")
        for name, param in model.named_parameters():
            clean_name = name.replace(".", "_")
            vals = param.detach().numpy().flatten().tolist()
            f.write(f"    const std::vector<float> {clean_name} = {{ {', '.join([f'{v:.8f}f' for v in vals])} }};\n")

        f.write("}\n")

    print(f"[SUCCESS] Ваги ({namespace}) успішно експортовано у {output_path}!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen", action="store_true",
                        help="експортувати генераторну модель (solve_nn_gen_output -> nn_weights_gen.h, namespace NNWeights_gen)")
    parser.add_argument("--outdir", default=None, help="каталог з моделлю")
    parser.add_argument("--output", default=None, help="шлях до вихідного header")
    parser.add_argument("--namespace", default=None, help="ім'я namespace у header")
    args = parser.parse_args()

    if args.gen:
        out_dir   = args.outdir    or "solve_nn_gen_output"
        output    = args.output    or "kinreco/nn_weights_gen.h"
        namespace = args.namespace or "NNWeights_gen"
    else:
        out_dir   = args.outdir    or "solve_nn_output"
        output    = args.output    or "kinreco/nn_weights.h"
        namespace = args.namespace or "NNWeights"

    create_weights(out_dir, output, namespace)
