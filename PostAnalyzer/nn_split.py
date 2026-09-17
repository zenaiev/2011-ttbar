"""
nn_split.py
───────────
Єдине джерело правди про те, які події бачила нейромережа LKRnn.

Модель у папці (solve_nn_output, solve_nn_gen_output, ...) — це два файли:
  solve_nn_best.pt    — ваги разом із нормуванням входу (буфери x_mean, x_std);
  dataset_split.root  — мітки датасету: для кожної придатної до тренування події (channel, entry)
                        позначка train / val / test, і службовий запис nn_split_info
                        (рівень тренування det/gen, межі бази LKRv3, частки поділу, сід).

  train_filter(a, level) — відбір подій, придатних до тренування (ним користується train_solve_nn).
  write_split / load_split — запис і читання міток.
  split_masks(...)       — маски train/val/test для файлу eventReco з перевіркою, що мітки
                           відповідають саме цим подіям і поточному фільтру.
  eval_sample(...)       — вибірка для оцінки LKRnn (події, яких мережа не бачила):
                             * події з міткою test;
                             * непридатні до тренування, узяті випадково з тією самою часткою.
                           Кожна подія потрапляє у вибірку з однаковою імовірністю незалежно від своїх
                           властивостей, тож це випадкова частина ПОВНОГО датасету, і LKRnn на ній
                           коректно порівнювати з алгоритмічними методами на всьому датасеті.
  uses_nn(method)        — чи метод використовує нейромережу. Для таких методів bias і роздільна
                           здатність рахуються лише на eval_sample. Ефективність — на всьому
                           датасеті: прапорець lkrnn збігається з lkrv3, тобто від ваг мережі не
                           залежить. LKR, LKRv2, LKRv3 — повністю на всьому датасеті.
"""

import functools
import json
import os

import numpy as np
import uproot

MODEL_DIRS = {"det": "solve_nn_output", "gen": "solve_nn_gen_output"}
SPLIT_FILE = "dataset_split.root"
LABELS = {"train": 0, "val": 1, "test": 2}
SPLIT_FRACTIONS = (0.70, 0.15)     # train, val; решта — test
SPLIT_SEED = 20260914              # сід для випадкової частки непридатних подій у eval_sample
MTT_BASE_RANGE = (0.0, 7000.0)     # база LKRv3: додатна (логарифм) і менша за √s; межі 300 ГеВ більше немає
APPLY_INFO_KEY = "nn_apply_info"   # службовий запис nn_apply.py: які моделі застосовано до det/gen-гілок


def uses_nn(method):
    """True для методів з нейромережею (lkrnn, lkrnn_gen) — лише їх обмежуємо eval_sample."""
    return method.lower().startswith("lkrnn")


def _names(level):
    sfx = "" if level == "det" else "_gen"
    return sfx, ("nn_features" if level == "det" else "nn_features_gen")


def filter_keys(level):
    """Гілки, потрібні для train_filter."""
    sfx, fbr = _names(level)
    keys = [fbr, f"lkrv3{sfx}", f"mtt_lkrv3{sfx}", f"pttt_lkrv3{sfx}", f"ytt_lkrv3{sfx}",
            "mtt_gen", "pttt_gen", "ytt_gen"]
    if level == "det":
        keys.append("reco_passed_selection")
    return keys


def train_filter(a, level):
    """Маска подій, придатних до тренування моделі рівня level ('det' або 'gen')."""
    sfx, fbr = _names(level)
    feats = a[fbr]
    f0 = (np.stack(feats) if feats.dtype == object else np.asarray(feats))[:, 0]
    mtt_lkr, pttt_lkr, ytt_lkr = a[f"mtt_lkrv3{sfx}"], a[f"pttt_lkrv3{sfx}"], a[f"ytt_lkrv3{sfx}"]
    mtt_gen, pttt_gen, ytt_gen = a["mtt_gen"], a["pttt_gen"], a["ytt_gen"]

    ok = (f0 != -999) & (a[f"lkrv3{sfx}"] == 1)          # заповнені ознаки й успішний LKRv3
    if level == "det":
        ok &= a["reco_passed_selection"] == 1               # лише події, що пройшли eventReco-відбір
    lo, hi = MTT_BASE_RANGE
    ok &= np.isfinite(mtt_lkr) & (mtt_lkr > lo) & (mtt_lkr < hi)
    ok &= np.isfinite(mtt_gen) & (mtt_gen > 0)
    # цілі мають бути скінченними (у тренуванні події з нефінітною ціллю відкидаються)
    with np.errstate(all="ignore"):
        finite = (np.isfinite(np.log(mtt_gen) - np.log(mtt_lkr))
                  & np.isfinite(np.log(pttt_gen + 1.0) - np.log(pttt_lkr + 1.0))
                  & np.isfinite(ytt_gen - ytt_lkr))
    return ok & finite


def split_file(model_dir):
    """Шлях до файлу міток моделі."""
    return os.path.join(model_dir, SPLIT_FILE)


def write_split(path, origin, idx_train, idx_val, idx_test, level, extra_meta=None):
    """Записує мітки датасету. origin — (N, 2): канал і номер запису ttbarTree кожної придатної події."""
    split = np.full(len(origin), -1, np.int8)
    split[idx_train], split[idx_val], split[idx_test] = LABELS["train"], LABELS["val"], LABELS["test"]
    if (split < 0).any():
        raise RuntimeError("не всі придатні події отримали мітку")
    ch, en = origin[:, 0].astype(np.int32), origin[:, 1].astype(np.int64)
    order = np.lexsort((en, ch))
    meta = {"level": level, "labels": LABELS, "mtt_base_range": list(MTT_BASE_RANGE),
            "fractions": list(SPLIT_FRACTIONS), "counts": {k: int((split == v).sum()) for k, v in LABELS.items()}}
    meta.update(extra_meta or {})
    with uproot.recreate(path) as f:
        f["nn_split"] = {"channel": ch[order], "entry": en[order], "split": split[order]}
        f["nn_split_info"] = json.dumps(meta)
    load_split.cache_clear()
    print(f"[I] Мітки датасету ({level}): {meta['counts']} -> {path}")


@functools.lru_cache(maxsize=8)
def load_split(path):
    """(мітки {channel, entry, split}, службовий запис) з dataset_split.root."""
    if not os.path.exists(path):
        raise RuntimeError(f"немає файлу міток {path} — модель не натренована або у старому форматі "
                           "(сконвертуйте: python convert_model_format.py --model-dir ... --level ...)")
    f = uproot.open(path)
    return f["nn_split"].arrays(library="np"), json.loads(str(f["nn_split_info"]))


def split_masks(path, fname, channel, treename="ttbarTree"):
    """Маски train/val/test для файлу eventReco fname (канал channel) з перевіркою відповідності.
    Повертає (рівень тренування, кількість подій, маска придатних, {'train', 'val', 'test'})."""
    labels, meta = load_split(path)
    level = meta["level"]
    if tuple(meta["mtt_base_range"]) != tuple(MTT_BASE_RANGE):
        raise RuntimeError(f"{path}: модель тренувалась з базою LKRv3 у межах {tuple(meta['mtt_base_range'])}, "
                           f"а поточний фільтр — {MTT_BASE_RANGE}; перетренуйте модель")
    tree = uproot.open(fname)[treename]
    n = tree.num_entries
    eligible = train_filter(tree.arrays(filter_keys(level), library="np"), level)
    sel = labels["channel"] == channel
    ent, lab = labels["entry"][sel], labels["split"][sel]
    if len(ent) and ent.max() >= n:
        raise RuntimeError(f"{path}: запис {ent.max()} поза межами {fname} ({n} подій) — мітки від інших файлів")
    masks = {}
    for name, code in LABELS.items():
        m = np.zeros(n, dtype=bool)
        m[ent[lab == code]] = True
        masks[name] = m
    labelled = masks["train"] | masks["val"] | masks["test"]
    if int(labelled.sum()) != len(ent) or not np.array_equal(labelled, eligible):
        raise RuntimeError(f"{path}: мітки не відповідають подіям у {fname} (позначено {len(ent)}, придатних "
                           f"{int(eligible.sum())}, розбіжностей {int((labelled != eligible).sum())}) — модель "
                           "натренована на інших файлах eventReco або з іншим фільтром")
    return level, n, eligible, masks


def apply_info(fname):
    """Словник з nn_apply_info, якщо файл створено nn_apply.py; інакше None (файл від C++)."""
    try:
        return json.loads(str(uproot.open(fname)[APPLY_INFO_KEY]))
    except (KeyError, ValueError, TypeError):
        return None


def eval_sample(fname, channel, model_level, split_path=None, seed=SPLIT_SEED, treename="ttbarTree"):
    """Маска подій файлу fname (канал channel), не використаних у тренуванні моделі model_level.
    Повертає (mask, info)."""
    # Файли від nn_apply.py знають, яка модель стоїть у det-/gen-гілці: беремо її мітки
    # (рівень тренування записано в них, тож для крос-тесту --split-model не потрібен).
    if split_path is None:
        info = apply_info(fname)
        model_dir = info[model_level]["model"] if (info and info.get(model_level)) else MODEL_DIRS[model_level]
        split_path = split_file(model_dir)
    level, n, eligible, masks = split_masks(split_path, fname, channel, treename)
    test = masks["test"]
    frac = test.sum() / eligible.sum() if eligible.sum() else 0.0
    rng = np.random.default_rng([seed, int(channel), 0 if level == "det" else 1])
    extra = ~eligible & (rng.random(n) < frac)
    info = {"split_file": split_path, "filter_level": level, "n": n, "eligible": int(eligible.sum()),
            "trained": int((masks["train"] | masks["val"]).sum()), "test": int(test.sum()),
            "extra": int(extra.sum()), "frac": float(frac)}
    return test | extra, info
