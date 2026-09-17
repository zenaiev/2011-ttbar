"""
convert_model_format.py
───────────────────────
Разова конвертація моделі зі старого формату в новий, без перетренування.

  старий: solve_nn_best.pt (лише шари) + norm_stats.json + nn_trained.root (train+val)
  новий:  solve_nn_best.pt (шари + буфери x_mean/x_std) + dataset_split.root (мітки train/val/test)

Мітки відновлюються точно: той самий фільтр, порядок подій і перестановка (SEED), що й у тренуванні.
Перед записом перевіряється, що:
  1) train+val з відновленого поділу точно збігаються з nn_trained.root;
  2) x_mean/x_std з norm_stats.json збігаються зі статистиками тренувальної частини;
  3) нова модель видає те саме, що стара модель з нормуванням, як у LKRnn.cxx.
Старі файли переносяться в <папка>_oldformat/.

  python convert_model_format.py --model-dir solve_nn_output --level det
  python convert_model_format.py --model-dir solve_nn_gen_output --level gen
"""

import argparse
import json
import os
import shutil

import numpy as np
import torch
import uproot

import nn_split
from train_solve_nn import SEED, SolveMLP, load_and_prepare


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", dest="model_dir", required=True)
    ap.add_argument("--level", choices=["det", "gen"], required=True, help="рівень, на якому тренувалась модель")
    ap.add_argument("--inputs", nargs="+",
                    default=["ttbar_output_full_1.root", "ttbar_output_full_2.root", "ttbar_output_full_3.root"],
                    help="файли eventReco, на яких тренувалась модель (порядок як у тренуванні)")
    args = ap.parse_args()

    d = args.model_dir.rstrip("/")
    pt, norm_json, trained_root = (os.path.join(d, x) for x in ("solve_nn_best.pt", "norm_stats.json", "nn_trained.root"))
    old_dir = d + "_oldformat"
    for p in (pt, norm_json, trained_root):
        if not os.path.exists(p):
            raise SystemExit(f"[E] немає {p}")
    if os.path.exists(old_dir):
        raise SystemExit(f"[E] {old_dir} уже існує — не перезаписую")
    state = torch.load(pt, map_location="cpu")
    if "x_mean" in state:
        raise SystemExit(f"[E] {pt} уже в новому форматі")
    norm = json.load(open(norm_json))
    xm_old, xs_old = np.asarray(norm["x_mean"], np.float32), np.asarray(norm["x_std"], np.float32)

    X, _, _, origin = load_and_prepare(args.inputs, "ttbarTree", args.level)
    N = len(X)
    idx = np.random.RandomState(SEED).permutation(N)
    n_tr, n_val = int(nn_split.SPLIT_FRACTIONS[0] * N), int(nn_split.SPLIT_FRACTIONS[1] * N)
    idx_tr, idx_val, idx_te = idx[:n_tr], idx[n_tr:n_tr + n_val], idx[n_tr + n_val:]

    # 1) train+val == nn_trained.root
    tr = uproot.open(f"{trained_root}:nn_trained").arrays(library="np")
    old_pairs = set(zip(tr["channel"].tolist(), tr["entry"].tolist()))
    new_pairs = set(map(tuple, origin[np.concatenate([idx_tr, idx_val])].tolist()))
    if old_pairs != new_pairs:
        raise SystemExit(f"[E] відновлений поділ не збігається з {trained_root}: "
                         f"{len(old_pairs ^ new_pairs)} розбіжностей — інші файли або інший фільтр")
    print(f"[OK] 1) train+val відновленого поділу = {trained_root} ({len(old_pairs)} подій)")

    # 2) нормування == статистики тренувальної частини
    dm = np.max(np.abs(X[idx_tr].mean(axis=0) - xm_old) / (np.abs(xm_old) + 1e-6))
    ds = np.max(np.abs((X[idx_tr].std(axis=0) + 1e-8) - xs_old) / xs_old)
    if dm > 1e-5 or ds > 1e-5:
        raise SystemExit(f"[E] norm_stats.json не відповідає тренувальній частині (mean {dm:.1e}, std {ds:.1e})")
    print(f"[OK] 2) norm_stats.json = статистики тренувальної частини (відн. різниця mean {dm:.1e}, std {ds:.1e})")

    # 3) нова модель == стара модель з нормуванням як у LKRnn.cxx
    model = SolveMLP()
    res = model.load_state_dict(state, strict=False)
    if set(res.missing_keys) != {"x_mean", "x_std"} or res.unexpected_keys:
        raise SystemExit(f"[E] неочікувана структура ваг: missing {res.missing_keys}, unexpected {res.unexpected_keys}")
    model.set_normalization(xm_old, xs_old)
    model.eval()
    ref = SolveMLP()
    ref.load_state_dict(state, strict=False)      # буфери за замовч. 0 / 1 -> нормування вручну нижче
    ref.eval()
    xb = X[idx_te[:50000]]
    with torch.no_grad():
        out_new = model(torch.from_numpy(xb)).numpy()
        out_ref = ref(torch.from_numpy(((xb - xm_old) / (xs_old + np.float32(1e-8))).astype(np.float32))).numpy()
    diff = np.max(np.abs(out_new - out_ref))
    if diff > 1e-6:
        raise SystemExit(f"[E] вихід нової моделі відрізняється від старої ({diff:.1e})")
    print(f"[OK] 3) вихід нової моделі = стара модель з нормуванням (макс. різниця {diff:.1e})")

    os.makedirs(old_dir)
    for p in (pt, norm_json, trained_root):
        shutil.move(p, os.path.join(old_dir, os.path.basename(p)))
    torch.save(model.state_dict(), pt)
    nn_split.write_split(nn_split.split_file(d), origin, idx_tr, idx_val, idx_te, args.level,
                         {"inputs": list(args.inputs), "seed": SEED,
                          "converted_from": "solve_nn_best.pt + norm_stats.json + nn_trained.root"})
    print(f"[I] {d}: solve_nn_best.pt (ваги + нормування), {nn_split.SPLIT_FILE}; старі файли -> {old_dir}/")


if __name__ == "__main__":
    main()
