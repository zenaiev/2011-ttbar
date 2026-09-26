"""
nn_apply.py
───────────
Тестовий режим LKRnn без C++: накладає ваги моделей (.pt) на готовий вихід eventReco
(ttbar_output_*_{ch}.root) і записує копію файлу з перерахованими гілками lkrnn / lkrnn_gen.

Вхід мережі (26 ознак nn_features / nn_features_gen) і база LKRv3 (mtt/pttt/ytt/phitt_lkrv3) уже є
у файлі, тож перезапускати ttbarMakeHist не треба. Обчислення повторює LKRnn.cxx крок у крок
(float32), тож результат збігається з C++ до ~1e-6 — перевіряється опцією --check.

Модель — це папка з двома файлами: solve_nn_best.pt (ваги + нормування входу) і dataset_split.root
(мітки train/val/test). У вихідний файл записується службовий об'єкт nn_apply_info (які моделі
застосовано до det- і gen-гілки та на якому рівні вони тренувались). nn_split.eval_sample бере
з нього мітки відповідної моделі, тому integral_res.py, paper_figs.py, kinreco_eff_v4.py і
correlation.py працюють з такими файлами без додаткових аргументів (і без --split-model для крос-тесту).

Запуск (середовище з torch, напр. `conda activate hep`):
  python nn_apply.py                                                 # det- і gen-модель за замовч.
  python nn_apply.py --det-model solve_nn_test --gen-model keep      # нова det-модель, gen-гілку не чіпати
  python nn_apply.py --det-model solve_nn_gen_output --gen-model keep \\
                     --output-pattern 'ttbar_output_nntest_gennn_{ch}.root'   # крос-тест gen-моделі
  python nn_apply.py --check 'ttbar_output_det_{ch}.root'            # звірити з гілками, порахованими C++
  python nn_apply.py --det-model keep --gen-model keep --study studies/models/mlp_raw_direct_l2:lkrnnmlpraw
  python nn_apply.py --det-model keep --gen-model keep \
      --study-cal studies/models/mlp_raw_direct_l2:lkrnnmlprawcal   # з калібруванням (calibration.json)
Далі, напр.:  python3 integral_res.py --input-pattern 'ttbar_output_nntest_{ch}.root'
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import uproot

import nn_split
from train_solve_nn import SolveMLP
import nn_extra
from train_study import calibrate, extra_kind, load_calibration, load_study, lkr_base, model_inputs, uses_extra

VARS = ("mtt", "ytt", "pttt", "phitt", "dphitt")
BRANCHES = {"det": ("", "nn_features"), "gen": ("_gen", "nn_features_gen")}   # гілка -> (суфікс, ознаки)
STEP = 200_000


def load_model(model_dir, device):
    """Модель з нормуванням усередині (буфери x_mean/x_std у solve_nn_best.pt)."""
    state = torch.load(os.path.join(model_dir, "solve_nn_best.pt"), map_location="cpu")
    if "x_mean" not in state:
        raise RuntimeError(f"{model_dir}/solve_nn_best.pt у старому форматі — сконвертуйте: "
                           f"python convert_model_format.py --model-dir {model_dir} --level det|gen")
    m = SolveMLP()
    m.load_state_dict(state)
    return m.to(device).eval()


def model_train_levels(model_dirs, files):
    """Рівень тренування кожної моделі з її міток; заодно перевіряє, що мітки відповідають цим файлам."""
    levels = {}
    for d in model_dirs:
        found = {nn_split.split_masks(nn_split.split_file(d), fname, ch)[0] for ch, fname in files}
        if len(found) != 1:
            raise RuntimeError(f"{d}: неоднозначний рівень тренування {found}")
        levels[d] = found.pop()
    return levels


def infer(model, feats, device, batch=65536):
    """Вихід мережі (з нормуванням, tanh і масштабом 0.05/0.20/0.05), як d_log_mtt, d_log_pttt, d_ytt у LKRnn.cxx."""
    x = np.ascontiguousarray(feats, dtype=np.float32)
    out = np.empty((len(x), 3), np.float32)
    with torch.no_grad():
        for i in range(0, len(x), batch):
            out[i:i + batch] = model(torch.from_numpy(x[i:i + batch]).to(device)).float().cpu().numpy()
    return out


def lkrnn_vars(d, has_jets, mtt3, pttt3, ytt3, phi3, base_floor=0.0):
    """Кроки 6–7 LKRnn.cxx: поправки до бази LKRv3 -> вектор tt̄ -> змінні, як у krvars.h (t = t̄ = tt̄/2).
    base_floor > 0 відтворює старий варіант коду з обрізанням бази: log(max(m_LKRv3, floor))."""
    f32 = np.float32
    with np.errstate(all="ignore"):
        if base_floor > 0:                                            # старий код: обрізання, без раннього return
            ok = has_jets.copy()
            base = np.maximum(mtt3.astype(f32), f32(base_floor))
        else:
            ok = has_jets & (mtt3 > 0)                                # if (!(mtt_lkrv3 > 0)) return
            base = np.where(ok, mtt3, 1).astype(f32)
        ytt_b = np.where(np.isfinite(ytt3), ytt3, 0).astype(f32)      # рапідність із захистом (E±Pz <= 0 -> 0)
        mtt = np.exp(np.log(base) + d[:, 0]).astype(f32)
        pttt = (np.exp(np.log(pttt3.astype(f32) + f32(1)) + d[:, 1]) - f32(1)).astype(f32)
        ytt = (ytt_b + d[:, 2]).astype(f32)
        ok &= ~np.isnan(mtt) & (mtt > 0) & ~np.isnan(pttt) & (pttt >= 0)
        phi = phi3.astype(f32)
        px, py = pttt * np.cos(phi), pttt * np.sin(phi)
        mt = np.sqrt(mtt * mtt + pttt * pttt)
        pz, e = mt * np.sinh(ytt), mt * np.cosh(ytt)
        px, py, pz, e = (v.astype(np.float64) for v in (px, py, pz, e))    # TLorentzVector — double
        mag2 = e * e - (px * px + py * py + pz * pz)
        out = {
            "mtt": np.where(mag2 < 0, -np.sqrt(-mag2), np.sqrt(mag2)),
            "ytt": 0.5 * np.log((e + pz) / (e - pz)),
            "pttt": np.sqrt(px * px + py * py),
            "phitt": np.where((px == 0) & (py == 0) & (pz == 0), 0.0, np.arctan2(py, px)),
            "dphitt": np.zeros(len(d)),                                 # t і t̄ однакові -> Δφ = 0
        }
    return ok, {k: v.astype(np.float32) for k, v in out.items()}


def study_corrections(pred, target, mtt3, pttt3, ytt3):
    """Прогноз дослідницької моделі -> поправки d у лог-просторі до бази LKRv3, щоб скористатися тією самою
    кінематикою lkrnn_vars (φ системи tt̄ — з LKRv3, тобто з суми лептонів, jets і MET). pT обрізається до >= 0."""
    f32 = np.float32
    with np.errstate(all="ignore"):
        log_m = np.log(np.where(mtt3 > 0, mtt3, 1)).astype(f32)
        log_p = np.log(pttt3.astype(f32) + f32(1))
        ytt_b = np.where(np.isfinite(ytt3), ytt3, 0).astype(f32)
        if target == "residual":
            d = pred.astype(f32).copy()
        else:
            d = np.stack([pred[:, 0] - log_m, pred[:, 1] - log_p, pred[:, 2] - ytt_b], axis=1).astype(f32)
        d[:, 1] = np.maximum(d[:, 1], -log_p + f32(1e-6))
    return np.nan_to_num(d)


def study_branch(chunk, out_name, sfx, smodel, cfg, cal, alpha, device):
    """Гілка дослідницької моделі на детекторних (sfx="") чи генераторних (sfx="_gen") входах."""
    feats = chunk["nn_features" if sfx == "" else "nn_features_gen"]
    has_jets = feats[:, 0] != -999
    if sfx == "":
        has_jets &= chunk["reco_passed_selection"] == 1
    if uses_extra(cfg) and sfx != "":
        raise SystemExit(f"[E] {out_name}: додаткові входи (nn_extra) є лише для детекторних подій")
    x = model_inputs(feats, cfg, chunk.get(f"_sc_{extra_kind(cfg)}") if uses_extra(cfg) else None)
    pred = np.zeros((len(feats), 3), np.float32)
    if has_jets.any():
        pred[has_jets] = infer(smodel, x[has_jets], device)
    m3, p3, y3, f3 = (chunk[f"{v}_lkrv3{sfx}"] for v in ("mtt", "pttt", "ytt", "phitt"))
    if cal is not None:                                 # numerical inversion: прибирає зсув у бінах справжньої величини
        pred = calibrate(pred, cal, alpha, lkr_base(m3, p3, y3))
    d = study_corrections(pred, cfg["target"], m3, p3, y3)
    ok, vals = lkrnn_vars(d, has_jets, m3, p3, y3, f3)
    bad3 = chunk[f"lkrv3{sfx}"] != 1
    chunk[out_name] = ok.astype(np.int32)
    for v in VARS:
        default = chunk[f"{v}_lkrv3{sfx}"][bad3][0] if bad3.any() else np.float32(-999.)
        chunk[f"{v}_{out_name}"] = np.where(ok, vals[v], default).astype(np.float32)


def process_channel(ch, fin, fout, models, info, device, studies=(), base_floor=0.0):
    tree = uproot.open(fin)["ttbarTree"]
    names = list(tree.keys())
    for sfx, _ in BRANCHES.values():
        for b in [f"lkrnn{sfx}"] + [f"{v}_lkrnn{sfx}" for v in VARS]:
            if b not in names:
                names.append(b)
    for name, _, _, _, _, level in studies:
        for sfx in ([""] + (["_gen"] if level == "gen" else [])):
            for b in [f"{name}{sfx}"] + [f"{v}_{name}{sfx}" for v in VARS]:
                if b not in names:
                    names.append(b)
    with uproot.recreate(fout, compression=uproot.ZLIB(1)) as out:
        out_tree = None
        kinds = {extra_kind(s[2]) for s in studies if uses_extra(s[2])}
        extra_all = {k: nn_extra.load(fin, ch, k) for k in kinds}
        offset = 0
        for chunk in tree.iterate(filter_name=[n for n in names if n in tree.keys()], library="np", step_size=STEP):
            n = len(chunk["mtt_gen"])
            for k, arr in extra_all.items():                            # супутники вирівняні із записами дерева
                chunk[f"_sc_{k}"] = arr[offset:offset + n]
            offset += n
            for branch, (sfx, fbr) in BRANCHES.items():
                if models[branch] is None:                              # keep: гілку не чіпаємо
                    continue
                feats = chunk[fbr]
                has_jets = feats[:, 0] != -999
                if branch == "det":
                    has_jets &= chunk["reco_passed_selection"] == 1
                d = infer(models[branch], feats, device)
                ok, vals = lkrnn_vars(d, has_jets, chunk[f"mtt_lkrv3{sfx}"], chunk[f"pttt_lkrv3{sfx}"],
                                      chunk[f"ytt_lkrv3{sfx}"], chunk[f"phitt_lkrv3{sfx}"], base_floor)
                bad3 = chunk[f"lkrv3{sfx}"] != 1                        # значення «немає розв'язку» — як у reset_vars
                chunk[f"lkrnn{sfx}"] = ok.astype(np.int32)
                for v in VARS:
                    default = chunk[f"{v}_lkrv3{sfx}"][bad3][0] if bad3.any() else np.float32(-999.)
                    chunk[f"{v}_lkrnn{sfx}"] = np.where(ok, vals[v], default).astype(np.float32)
            for name, smodel, cfg, cal, alpha, level in studies:   # дослідницькі моделі (train_study.py)
                # детекторні входи -> гілка {name}: det–det для det-моделі, крос-тест gen→det для gen-моделі;
                # генераторні входи -> гілка {name}_gen (лише gen-модель: gen–gen)
                for sfx in ([""] + (["_gen"] if level == "gen" else [])):
                    study_branch(chunk, f"{name}{sfx}", sfx, smodel, cfg, cal, alpha, device)
            if out_tree is None:
                types = {k: (chunk[k].dtype if chunk[k].ndim == 1 else np.dtype((chunk[k].dtype, chunk[k].shape[1:])))
                         for k in names}
                out_tree = out.mktree("ttbarTree", types)
            out_tree.extend({k: chunk[k] for k in names})
        out["nn_apply_info"] = json.dumps(info)


def check_channel(ch, fnew, fref):
    """Порівняння гілок lkrnn з еталонним файлом (напр. порахованим C++)."""
    keys = [f"{p}{sfx}" for sfx, _ in BRANCHES.values() for p in ["lkrnn"] + [f"{v}_lkrnn" for v in VARS]]
    a = uproot.open(fnew)["ttbarTree"].arrays(keys, library="np")
    b = uproot.open(fref)["ttbarTree"].arrays(keys, library="np")
    for branch, (sfx, _) in BRANCHES.items():
        fa, fb = a[f"lkrnn{sfx}"] == 1, b[f"lkrnn{sfx}"] == 1
        both = fa & fb
        rel = lambda v: np.max(np.abs(a[v][both] - b[v][both]) / np.maximum(np.abs(b[v][both]), 1e-3)) if both.any() else 0.0
        ab = lambda v: np.max(np.abs(a[v][both] - b[v][both])) if both.any() else 0.0
        print(f"    канал {ch}, {branch}-гілка: розв'язків {fa.sum()} / {fb.sum()}, прапорці різні: {(fa != fb).sum()}; "
              f"макс. відн. різниця M {rel(f'mtt_lkrnn{sfx}'):.1e}, pT {rel(f'pttt_lkrnn{sfx}'):.1e}; "
              f"макс. абс. різниця y {ab(f'ytt_lkrnn{sfx}'):.1e}, φ {ab(f'phitt_lkrnn{sfx}'):.1e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-pattern", dest="input_pattern", default="ttbar_output_det_{ch}.root",
                    help="вихід eventReco з ознаками й базою LKRv3 (будь-який набір: det або full)")
    ap.add_argument("--output-pattern", dest="output_pattern", default="ttbar_output_nntest_{ch}.root")
    ap.add_argument("--channels", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--det-model", dest="det_model", default="solve_nn_output",
                    help="модель для детекторної гілки lkrnn (папка з solve_nn_best.pt і dataset_split.root) або keep")
    ap.add_argument("--gen-model", dest="gen_model", default="solve_nn_gen_output",
                    help="модель для генераторної гілки lkrnn_gen або keep")
    ap.add_argument("--study", action="append", default=[], metavar="DIR:NAME",
                    help="дослідницька модель train_study.py і назва її гілок (починається з lkrnn), "
                         "напр. studies/models/mlp_raw_direct_l2:lkrnnmlpraw; можна кілька разів")
    ap.add_argument("--study-cal", dest="study_cal", action="append", default=[], metavar="DIR:NAME",
                    help="те саме, але з калібруванням (calibration.json з папки моделі): прогноз проганяється "
                         "через обернену середню відповідь, що прибирає зсув у бінах справжньої величини ціною "
                         "розкиду; ПАПКА:НАЗВА:α задає часткове калібрування (α=0.5 — половина поправки)")
    ap.add_argument("--check", default=None, metavar="PATTERN",
                    help="звірити отримані гілки lkrnn з цими файлами (напр. порахованими C++ з тими самими моделями)")
    ap.add_argument("--base-floor", dest="base_floor", type=float, default=0.0, metavar="ГеВ",
                    help="відтворити старий код із обрізанням бази LKRv3: log(max(m, ГеВ)); 300 — стан до коміту 274496f")
    args = ap.parse_args()

    files = [(ch, args.input_pattern.format(ch=ch)) for ch in args.channels]
    for ch, f in files:
        if not os.path.exists(f):
            raise SystemExit(f"[E] немає вхідного файлу {f}")
        if os.path.abspath(f) == os.path.abspath(args.output_pattern.format(ch=ch)):
            raise SystemExit("[E] вихідний файл збігається з вхідним")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dirs = {"det": None if args.det_model == "keep" else args.det_model,
            "gen": None if args.gen_model == "keep" else args.gen_model}
    used = sorted({d for d in dirs.values() if d})
    t0 = time.time()
    levels = model_train_levels(used, files)
    models = {b: (load_model(d, device) if d else None) for b, d in dirs.items()}
    info = {b: ({"model": os.path.abspath(d), "train_level": levels[d]} if d else None) for b, d in dirs.items()}
    info["input_pattern"] = args.input_pattern
    # вхід сам від nn_apply.py: гілки, яких цей запуск не чіпає, копіюються — переносимо і їхній опис
    prev = nn_split.apply_info(files[0][1])
    if prev:
        for b in ("det", "gen"):
            if dirs[b] is None and prev.get(b):
                info[b] = prev[b]
        if prev.get("studies"):
            info["studies"] = dict(prev["studies"])
        info["chained_from"] = prev.get("input_pattern")
        print(f"[I] вхід уже оброблений nn_apply.py: переношу опис {len(prev.get('studies') or {})} дослідницьких гілок")
    if args.base_floor > 0:
        info["base_floor"] = args.base_floor
        print(f"[I] база LKRv3 обрізається знизу на {args.base_floor:g} ГеВ (старий варіант LKRnn.cxx)")
    for b, d in dirs.items():
        print(f"[I] {b}-гілка: " + (f"модель {d} (тренована на {levels[d]}-рівні)" if d else "без змін (keep)"))
    studies = []
    for spec, calibrated in [(s, False) for s in args.study] + [(s, True) for s in args.study_cal]:
        alpha = 1.0
        if calibrated:                                      # ПАПКА:НАЗВА:α — часткове калібрування
            head, _, tail = spec.rpartition(":")
            try:
                alpha, spec = float(tail), head
            except ValueError:
                pass
        sdir, _, name = spec.rpartition(":")
        if not sdir or not name.startswith("lkrnn") or name in ("lkrnn", "lkrnn_gen") or not name.isidentifier():
            raise SystemExit(f"[E] --study {spec}: формат ПАПКА:НАЗВА, назва на кшталт lkrnnmlpraw")
        smodel, scfg = load_study(sdir, device)
        la, meta = nn_split.load_split(nn_split.split_file(sdir))
        level = meta["level"]
        # тестові події дослідницької моделі мають збігатися з тими, на яких оцінюється LKRnn того ж рівня
        ref_dir = dirs[level] or nn_split.MODEL_DIRS[level]
        lb, _ = nn_split.load_split(nn_split.split_file(ref_dir))
        if not all(np.array_equal(la[k], lb[k]) for k in ("channel", "entry", "split")):
            raise SystemExit(f"[E] {sdir}: поділ датасету не збігається з {ref_dir} — оцінка була б некоректною")
        for ch, f in files:                                 # мітки мають відповідати подіям цих файлів
            nn_split.split_masks(nn_split.split_file(sdir), f, ch)
        cal = None
        if calibrated:
            cal = load_calibration(sdir)
            if cal is None:
                raise SystemExit(f"[E] {sdir}: немає calibration.json (перетренуйте модель актуальним train_study.py)")
            if cal.get("target") != scfg["target"]:
                raise SystemExit(f"[E] {sdir}: калібрування для цілі {cal.get('target')}, а модель — {scfg['target']}")
        studies.append((name, smodel, scfg, cal, alpha, level))
        info.setdefault("studies", {})[name] = {"model": os.path.abspath(sdir), "train_level": level,
                                                "model_type": scfg["model"], "flat": scfg.get("flat", []),
                                                "calibrated": calibrated, "cal_alpha": alpha if calibrated else None,
                                                "estimator": scfg.get("estimator"), "augment_z": scfg.get("augment_z"), "bound": scfg.get("bound"),
                                                **{k: scfg[k] for k in ("inputs", "target", "loss")}}
        print(f"[I] {name}: {scfg['model']} ({level}-рівень"
              + (f"; гілки {name}_gen — gen–gen, {name} — крос-тест gen→det" if level == "gen" else "") + "), "
              f"вхід {scfg['inputs']}, ціль {scfg['target']}, лос {scfg['loss']}"
              f"{f', калібровано (α={alpha:g})' if calibrated else ''} ({sdir})")
    print(f"[I] пристрій: {device}")

    for ch, fin in files:
        fout = args.output_pattern.format(ch=ch)
        t1 = time.time()
        process_channel(ch, fin, fout, models, info, device, studies, args.base_floor)
        print(f"[I] канал {ch}: {fin} -> {fout}  ({time.time() - t1:.0f} с)")
        if args.check:
            check_channel(ch, fout, args.check.format(ch=ch))
    print(f"[I] готово за {time.time() - t0:.0f} с")


if __name__ == "__main__":
    main()
