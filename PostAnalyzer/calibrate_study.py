"""
calibrate_study.py
──────────────────
Калібрування вже натренованої дослідницької моделі (train_study.py) без перетренування:
будує калібрувальні криві «середній прогноз як функція справжнього значення» у просторі цілі
(log M, log(pT+1), y або поправки до LKRv3) на подіях train (за замовч.; val лишається для вибору α, тест не чіпається) —
і кладе calibration.json у папку моделі. nn_apply.py --study-cal обертає ці криві, що прибирає
зсув у бінах справжньої величини (той самий принцип, що в калібруванні відгуку jets).

Потрібне лише для моделей, натренованих до появи калібрування: нові train_study.py пише
calibration.json сам.

  python calibrate_study.py --model-dir studies/models/mlp_raw_direct_l2 studies/models/mlp_raw_direct_l2_flatM
"""

import argparse
import json
import os

import numpy as np
import torch

import nn_split
from train_solve_nn import SEED, load_and_prepare
from train_study import (cal_file, extra_for_origin, extra_kind, fit_calibration, load_study, make_targets, model_inputs,
                         uses_extra)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", dest="model_dirs", nargs="+", required=True)
    ap.add_argument("--data", nargs="+",
                    default=["ttbar_output_full_1.root", "ttbar_output_full_2.root", "ttbar_output_full_3.root"],
                    help="ті самі файли eventReco, на яких модель тренувалася")
    ap.add_argument("--force", action="store_true", help="перезаписати наявний calibration.json")
    ap.add_argument("--fit-on", dest="fit_on", choices=["train", "trainval"], default="train",
                    help="на яких подіях будувати криві; train — щоб val лишився для вибору α (scan_alpha.py)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = []
    for d in args.model_dirs:
        out = cal_file(d)                  # для flow: ПАПКА@оцінка -> calibration_<оцінка>.json
        if os.path.exists(out) and not args.force:
            print(f"[I] {out} уже є — пропускаю (--force, щоб перезаписати)")
            continue
        models.append((d, *load_study(d, device)))
    if not models:
        return
    levels = {cfg["level"] for _, _, cfg in models}
    if len(levels) != 1:
        raise SystemExit(f"[E] моделі різних рівнів {levels} — калібруйте окремо")

    X_all, y_res, lkr, origin = load_and_prepare(args.data, "ttbarTree", levels.pop())
    N = len(X_all)
    # той самий поділ, що в train_study.main (детермінований за SEED і тими самими файлами)
    idx = np.random.RandomState(SEED).permutation(N)
    n_tr, n_val = int(nn_split.SPLIT_FRACTIONS[0] * N), int(nn_split.SPLIT_FRACTIONS[1] * N)
    rows = idx[:n_tr] if args.fit_on == "train" else idx[:n_tr + n_val]
    print(f"[I] {N} придатних подій, калібрування на {args.fit_on} ({len(rows)})")

    for d, model, cfg in models:
        X = model_inputs(X_all, cfg, extra_for_origin(args.data, origin, extra_kind(cfg)) if uses_extra(cfg) else None)
        if X.shape[1] != cfg["in_dim"]:
            raise SystemExit(f"[E] {d}: модель чекає {cfg['in_dim']} входів, а дані дають {X.shape[1]}")
        Y = make_targets(y_res, lkr, cfg["target"])
        base = make_targets(np.zeros_like(y_res), lkr, "direct") if cfg["target"] == "residual" else None
        cal = fit_calibration(model, X, Y, rows, device, cfg["target"], fit_on=args.fit_on, base=base)
        with open(cal_file(d), "w") as f:
            json.dump(cal, f, indent=1)
        d0 = cal["dims"][0]
        slope = (d0["pred"][-1] - d0["pred"][0]) / (d0["true"][-1] - d0["true"][0])
        print(f"[I] {d}: нахил відповіді за log M {slope:.3f} (1.0 = без стискання) -> {cal_file(d)}")


if __name__ == "__main__":
    main()
