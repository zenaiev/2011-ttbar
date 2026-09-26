"""
nn_extra.py
───────────
Файл-«супутник» із додатковими входами дослідницьких мереж — без зміни eventReco і перезапуску ttbarMakeHist.
Величини беруться прямо з ntuples сигнального MC й вирівнюються з деревами ttbar_output_* за номером запису:
у дерево eventReco (StoreAllVars) потрапляють усі події свого каналу (mcEventType == канал) у порядку ланцюжка
ntuples, тож k-й запис дерева — k-та така подія. Вирівнювання перевіряється за mtt_gen (з mcT + mcTbar).

Jets відбираються як у eventReco.h: |η| <= 2.4, енергія без мюонної й електронної частки, pT > 30 ГеВ.
Два jets, вибрані LKRv3 (nn_features), знаходяться серед них за (px, py, pz) — їх не змінює мітка b-тегу.

Гілки супутника ttbar_output_extra_{ch}.root (дерево extraTree), для подій без LKRv3-jets — -999:
  nn_extra[10]    входи мереж (--inputs raw_extra / features_extra у train_study.py):
                    0, 1  дискримінант CSV LKRv3-jets 1 і 2 (повне значення, не лише «так/ні»);
                    2     кількість jets;  3  кількість b-тегованих (CSV > 0.244);  4  кількість первинних вершин;
                    5     найбільший CSV серед решти jets (-2, якщо їх немає);
                    6–9   сумарний 4-вектор решти jets E, px, py, pz (нулі, якщо їх немає);
  n_all_jets, all_jets[10][5] (E, px, py, pz, CSV), best_jets_idx[2]  — усі jets і індекси вибраних;
  mcT, mcTbar, mcNu, mcNubar (px, py, pz, m)                            — генераторні t, t̄, ν, ν̄;
  mtt_gen                                                               — для перевірки вирівнювання.

  python nn_extra.py --ref 'ttbar_output_full_{ch}.root'
"""

import argparse
import os
import re

import awkward as ak
import numpy as np
import uproot

N_EXTRA = 10
ROT_PX_EXTRA = [7]           # індекси px (py = px + 1) серед nn_extra — для азимутальної аугментації
PZ_FLIP_EXTRA = [9]          # pz — для відображення z -> −z
MAX_JETS = 10
BTAG_L = 0.244               # bTagDiscrL у eventReco.h
SIG_DIR = "ntuples-mc/TTJets_TuneZ2_7TeV-madgraph-tauola"
SIG_PARTS = ["00001", "010000", "010003", "010002", "010001", "00000"]   # порядок AddToChain у ttbarMakeHist.cxx
BRANCHES = ["mcEventType", "Npv", "jetPt", "jetEta", "jetPhi", "jetMass", "jetMuEn", "jetElEn", "jetBTagDiscr",
            "mcT", "mcTbar", "mcNu", "mcNubar"]


def sidecar_path(fname, ch):
    """ttbar_output_<що завгодно>_{ch}.root -> ttbar_output_extra_{ch}.root у тій самій папці."""
    return os.path.join(os.path.dirname(fname), f"ttbar_output_extra_{ch}.root")


def channel_of(fname):
    m = re.search(r"_(\d+)\.root$", fname)
    return int(m.group(1)) if m else None


def pair_mass(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    e = np.sqrt((a[:, :3] ** 2).sum(1) + a[:, 3] ** 2) + np.sqrt((b[:, :3] ** 2).sum(1) + b[:, 3] ** 2)
    p = a[:, :3] + b[:, :3]
    return np.sqrt(np.maximum(e ** 2 - (p ** 2).sum(1), 0))


def read_signal(ch, base="."):
    """Події каналу ch з ntuples сигналу в порядку ланцюжка eventReco."""
    parts = []
    for d in SIG_PARTS:
        f = os.path.join(base, SIG_DIR, d, "ttbarSel_merged.root")
        for chunk in uproot.open(f)["tree"].iterate(BRANCHES, library="ak", step_size="200 MB"):
            parts.append(chunk[chunk["mcEventType"] == ch])
    return ak.concatenate(parts)


def build(ch, ref, base="."):
    ev = read_signal(ch, base)
    t = uproot.open(ref)["ttbarTree"]
    r = t.arrays(["nn_features", "reco_passed_selection", "mtt_gen"], library="np")
    F = r["nn_features"]
    F = (np.stack(F) if F.dtype == object else F).astype(np.float64)
    N = len(F)
    if len(ev) != N:
        raise SystemExit(f"[E] канал {ch}: у ntuples {len(ev)} подій каналу, у {ref} — {N}")
    mcT, mcTbar = ak.to_numpy(ev["mcT"]), ak.to_numpy(ev["mcTbar"])
    dm = np.abs(pair_mass(mcT, mcTbar) - r["mtt_gen"]) / np.maximum(r["mtt_gen"], 1)
    if dm.max() > 1e-4:
        raise SystemExit(f"[E] канал {ch}: mtt_gen не збігається (макс. відн. {dm.max():.1e}) — порядок подій інший")

    # jets як у eventReco.h: SetPtEtaPhiM, мінус мюонна й електронна енергія, pT > 30, |η| <= 2.4
    f64 = lambda b: ak.values_astype(ev[b], np.float64)          # як у C++: double з float-входів
    pt, eta, phi, m = f64("jetPt"), f64("jetEta"), f64("jetPhi"), f64("jetMass")
    p2 = (pt * np.cosh(eta)) ** 2
    e = np.sqrt(np.maximum(p2 + np.sign(m) * m ** 2, 0))
    corr_e = e - f64("jetMuEn") - f64("jetElEn")
    corr_pt = pt * corr_e / e
    keep = (abs(eta) <= 2.4) & (corr_pt >= 30.0)
    jpt, jeta, jphi, je, jcsv = corr_pt[keep], eta[keep], phi[keep], corr_e[keep], ev["jetBTagDiscr"][keep]
    jpx, jpy, jpz = jpt * np.cos(jphi), jpt * np.sin(jphi), jpt * np.sinh(jeta)

    has = (r["reco_passed_selection"] == 1) & (F[:, 0] != -999)
    nj = ak.to_numpy(ak.num(jpt))
    # LKRv3-jets (nn_features: j1 = [8:12], j2 = [12:16], E px py pz) серед відібраних
    idx = np.full((N, 2), -1, np.int32)
    for k, off in ((0, 8), (1, 12)):
        d = (abs(jpx - F[:, off + 1]) + abs(jpy - F[:, off + 2]) + abs(jpz - F[:, off + 3]))
        am = ak.to_numpy(ak.fill_none(ak.argmin(d, axis=1), -1))
        dmin = ak.to_numpy(ak.fill_none(ak.min(d, axis=1), np.inf))
        bad = has & (dmin > 0.01 + 1e-4 * np.abs(F[:, off + 1:off + 4]).sum(1))    # float32 у nn_features
        if bad.any():
            raise SystemExit(f"[E] канал {ch}: jet{k + 1} LKRv3 не знайдено серед відібраних у {int(bad.sum())} подіях")
        idx[has, k] = am[has]

    # фіксовані масиви (N, MAX_JETS)
    def pad(x, fill=-999.0):
        return ak.to_numpy(ak.fill_none(ak.pad_none(x, MAX_JETS, clip=True), fill)).astype(np.float64)
    E, PX, PY, PZ, CSV = (pad(x) for x in (je, jpx, jpy, jpz, jcsv))
    slot = np.arange(MAX_JETS)[None, :]
    is_best = (slot == idx[:, :1]) | (slot == idx[:, 1:])
    is_jet = slot < nj[:, None]
    other = is_jet & ~is_best
    n_btag = ak.to_numpy(ak.sum(jcsv > BTAG_L, axis=1))

    extra = np.full((N, N_EXTRA), -999.0)
    rows = np.nonzero(has)[0]
    extra[rows, 0] = CSV[rows, idx[rows, 0]]
    extra[rows, 1] = CSV[rows, idx[rows, 1]]
    extra[rows, 2] = nj[rows]
    extra[rows, 3] = n_btag[rows]
    extra[rows, 4] = ak.to_numpy(ev["Npv"])[rows]
    extra[rows, 5] = np.where(other[rows].any(1), np.where(other[rows], CSV[rows], -np.inf).max(1), -2.0)
    for col, X in ((6, E), (7, PX), (8, PY), (9, PZ)):
        extra[rows, col] = np.where(other[rows], X[rows], 0.0).sum(1)
    if nj[has].max() > MAX_JETS:
        print(f"[W] канал {ch}: {int((nj[has] > MAX_JETS).sum())} подій мають > {MAX_JETS} jets — у all_jets лише перші "
              f"{MAX_JETS} (у nn_extra враховано всі)")
        # сума й максимум решти jets для таких подій — з повного списку
        many = np.nonzero(has & (nj > MAX_JETS))[0]
        for i in many:
            sel = [j for j in range(nj[i]) if j not in idx[i]]
            extra[i, 5] = max(float(jcsv[i][j]) for j in sel)
            for col, X in ((6, je), (7, jpx), (8, jpy), (9, jpz)):
                extra[i, col] = sum(float(X[i][j]) for j in sel)

    all_jets = np.stack([E, PX, PY, PZ, CSV], axis=2)
    all_jets[~has] = -999.0
    out = {"nn_extra": extra.astype(np.float32),
           "n_all_jets": np.where(has, np.minimum(nj, MAX_JETS), 0).astype(np.int32),
           "all_jets": all_jets.astype(np.float32),
           "best_jets_idx": idx,
           "mcT": mcT.astype(np.float32), "mcTbar": mcTbar.astype(np.float32),
           "mcNu": ak.to_numpy(ev["mcNu"]).astype(np.float32), "mcNubar": ak.to_numpy(ev["mcNubar"]).astype(np.float32),
           "mtt_gen": r["mtt_gen"].astype(np.float32)}
    return out, has


_cache = {}


def load(fname, ch=None, kind="extra"):
    """Входи із супутника, вирівняні із записами дерева fname (будь-який ttbar_output_*_{ch}.root тих самих подій):
    kind="extra" — nn_extra (N, 10) з цього скрипта; kind="pair" — pair_raw + pair_extra (N, 30) з pair_select.py."""
    ch = ch if ch is not None else channel_of(fname)
    if kind == "extra":
        path, tree, cols, how = sidecar_path(fname, ch), "extraTree", ["nn_extra"], "python nn_extra.py"
    else:
        path = os.path.join(os.path.dirname(fname), f"ttbar_output_pairsel_{ch}.root")
        tree, cols, how = "pairTree", ["pair_raw", "pair_extra"], "python pair_select.py apply"
    if not os.path.exists(path):
        raise SystemExit(f"[E] немає {path} — створіть: {how}")
    key = (os.path.abspath(path), os.path.abspath(fname))
    if key not in _cache:
        s = uproot.open(path)[tree].arrays(cols + ["mtt_gen"], library="np")
        g = uproot.open(fname)["ttbarTree"]["mtt_gen"].array(library="np")
        if len(g) != len(s["mtt_gen"]) or not np.allclose(g, s["mtt_gen"], rtol=1e-6):
            raise SystemExit(f"[E] {path} не вирівняний із {fname} (інші події)")
        _cache[key] = np.concatenate([s[c] for c in cols], axis=1).astype(np.float32)
    return _cache[key]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", default="ttbar_output_full_{ch}.root", help="дерево eventReco, з яким вирівнювати")
    ap.add_argument("--channels", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--base", default=".", help="папка, у якій лежить ntuples-mc")
    args = ap.parse_args()
    for ch in args.channels:
        ref = args.ref.format(ch=ch)
        out, has = build(ch, ref, args.base)
        path = sidecar_path(ref, ch)
        with uproot.recreate(path) as f:
            f.mktree("extraTree", {k: (v.dtype if v.ndim == 1 else np.dtype((v.dtype, v.shape[1:]))) for k, v in out.items()})
            f["extraTree"].extend(out)
        x = out["nn_extra"][has]
        print(f"[I] канал {ch}: {len(has)} подій, з LKRv3-jets {int(has.sum())} -> {path}")
        print(f"    CSV вибраних (медіана) {np.median(x[:, 0]):.3f} / {np.median(x[:, 1]):.3f}; "
              f"jets: {np.bincount(x[:, 2].astype(int))[2:].tolist()} (від 2); b-тег: {np.bincount(x[:, 3].astype(int)).tolist()}; "
              f"вершин (медіана) {np.median(x[:, 4]):.0f}; з іншими jets {100 * np.mean(x[:, 5] > -2):.1f}%")


if __name__ == "__main__":
    main()
