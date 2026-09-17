"""
export_onnx.py
──────────────
Експорт моделі з папки (solve_nn_best.pt: ваги + нормування входу) у solve_nn.onnx, який C++ (LKRnn)
читає під час запуску через ROOT TMVA SOFIE. Тренування робить це автоматично; скрипт — для вже
натренованих моделей. Після експорту вихід ONNX звіряється з PyTorch.

  python export_onnx.py --model-dir solve_nn_output solve_nn_gen_output
"""

import argparse
import os

import numpy as np
import torch
from onnx.reference import ReferenceEvaluator

from train_solve_nn import ONNX_NAME, SolveMLP, export_onnx


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", dest="model_dir", nargs="+", required=True)
    args = ap.parse_args()

    for d in args.model_dir:
        pt = os.path.join(d, "solve_nn_best.pt")
        state = torch.load(pt, map_location="cpu")
        if "x_mean" not in state:
            raise SystemExit(f"[E] {pt} у старому форматі — спершу: python convert_model_format.py --model-dir {d} --level det|gen")
        path = os.path.join(d, ONNX_NAME)
        export_onnx(state, path)

        m = SolveMLP()
        m.load_state_dict(state)
        m.eval()
        rng = np.random.default_rng(1)
        x = (m.x_mean.numpy() + m.x_std.numpy() * rng.standard_normal((200, 26))).astype(np.float32)
        with torch.no_grad():
            ref = m(torch.from_numpy(x)).numpy()
        sess = ReferenceEvaluator(path)
        got = np.concatenate([sess.run(None, {"x": x[i:i + 1]})[0] for i in range(len(x))])
        diff = float(np.max(np.abs(got - ref)))
        if diff > 1e-5:
            raise SystemExit(f"[E] {path}: вихід ONNX відрізняється від PyTorch ({diff:.1e})")
        print(f"[OK] {path}: вихід ONNX = PyTorch на 200 входах (макс. різниця {diff:.1e})")


if __name__ == "__main__":
    main()
