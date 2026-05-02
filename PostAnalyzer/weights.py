import torch
import json
import os
import sys
sys.path.append(os.getcwd())
from train_solve_nn import SolveMLP

def create_weights():
    out_dir = "solve_nn_output"

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

    output_path = "kinreco/nn_weights.h"
    with open(output_path, "w") as f:
        f.write("#pragma once\n#include <vector>\n\nnamespace NNWeights {\n")

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

    print(f"[SUCCESS] Ваги успішно експортовано у {output_path}!")

if __name__ == "__main__":
    create_weights()