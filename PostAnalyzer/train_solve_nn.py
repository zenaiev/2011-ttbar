import argparse, json, os, time
import concurrent.futures
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import uproot

# --- ЗОЛОТИЙ СТАНДАРТ (Швидкість GPU + Weighted L1 Loss) ---
IN_DIM       = 26     # Абсолютна сліпота
OUT_DIM      = 3      
BATCH_SIZE   = 4096   
LR           = 5e-4
WEIGHT_DECAY = 1e-2   
PATIENCE     = 50       
EPOCHS       = 800
SEED         = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    # Оптимізація для відеокарт NVIDIA
    torch.backends.cudnn.benchmark = True

def invariant_mass_2body(pt1, eta1, phi1, e1, pt2, eta2, phi2, e2):
    px = pt1*np.cos(phi1) + pt2*np.cos(phi2)
    py = pt1*np.sin(phi1) + pt2*np.sin(phi2)
    pz = pt1*np.sinh(eta1) + pt2*np.sinh(eta2)
    e  = e1 + e2
    return np.sqrt(np.maximum(e*e - px*px - py*py - pz*pz, 0.0))

def select_best_jets_event(lepM, lepP, jets, jet_masses):
    n = len(jets)
    best_bTag, best_pTSum, found = 0, 0.0, False
    jb1 = jb2 = None
    for i in range(n):
        j1 = jets[i]
        flagLepM1 = invariant_mass_2body(j1["pt"],j1["eta"],j1["phi"],j1["e"], lepM["pt"],lepM["eta"],lepM["phi"],lepM["e"]) < 180.
        flagLepP1 = invariant_mass_2body(j1["pt"],j1["eta"],j1["phi"],j1["e"], lepP["pt"],lepP["eta"],lepP["phi"],lepP["e"]) < 180.
        if not (flagLepM1 or flagLepP1): continue
        for k in range(n):
            if i == k: continue
            j2 = jets[k]
            flagLepM2 = invariant_mass_2body(j2["pt"],j2["eta"],j2["phi"],j2["e"], lepM["pt"],lepM["eta"],lepM["phi"],lepM["e"]) < 180.
            flagLepP2 = invariant_mass_2body(j2["pt"],j2["eta"],j2["phi"],j2["e"], lepP["pt"],lepP["eta"],lepP["phi"],lepP["e"]) < 180.
            if not (flagLepM2 or flagLepP2): continue
            nTrue = int(flagLepM1)+int(flagLepP1)+int(flagLepM2)+int(flagLepP2)
            if nTrue == 2:
                if ((flagLepM1 and flagLepP1) or (flagLepM2 and flagLepP2) or (flagLepM1 and flagLepM2) or (flagLepP1 and flagLepP2)): continue
            bTag = int(jet_masses[i] < 0) + int(jet_masses[k] < 0)
            if bTag < best_bTag: continue
            if bTag > best_bTag: best_pTSum = 0.0
            best_bTag = bTag
            pTSum = j1["pt"] + j2["pt"]
            if pTSum < best_pTSum: continue
            best_pTSum = pTSum
            def fix(j, m):
                if m < 0:
                    pt,eta,phi,mpos = j["pt"],j["eta"],j["phi"],-m
                    px,py,pz = pt*np.cos(phi), pt*np.sin(phi), pt*np.sinh(eta)
                    return {"pt":pt,"eta":eta,"phi":phi,"e":np.sqrt(px*px+py*py+pz*pz+mpos*mpos)}
                return j
            jb1, jb2, found = fix(j1,jet_masses[i]), fix(j2,jet_masses[k]), True
    return (jb1, jb2) if found else (None, None)

def ttbar_vars(t_e, t_px, t_py, t_pz, tb_e, tb_px, tb_py, tb_pz, lkr_vars):
    tt_e  = t_e  + tb_e
    tt_px = t_px + tb_px
    tt_py = t_py + tb_py
    tt_pz = t_pz + tb_pz

    mtt2 = tt_e**2 - tt_px**2 - tt_py**2 - tt_pz**2
    mtt  = np.sqrt(max(mtt2, 0.0))
    if mtt < 300.0: return np.array([np.nan]*3, dtype=np.float32)

    pttt  = np.sqrt(tt_px**2 + tt_py**2)
    ep, em = tt_e + tt_pz, tt_e - tt_pz
    ytt = 0.5 * np.log(ep / em) if (ep > 0 and em > 0) else 0.0

    d_log_mtt  = np.log(mtt) - lkr_vars[0]
    d_log_pttt = np.log(pttt + 1.0) - lkr_vars[1]
    d_ytt      = ytt - lkr_vars[2]

    return np.array([d_log_mtt, d_log_pttt, d_ytt], dtype=np.float32)

def process_single_batch(args):
    filename, treename, start, stop = args
    import uproot
    import awkward as ak
    import numpy as np
    
    MASS_MU, MASS_EL = 0.105658, 0.000511
    def calc_e(pt, eta, mass): return np.sqrt(pt**2 + (pt*np.sinh(eta))**2 + mass**2)

    branches = ["Nmu", "Nel", "Njet", "muPt", "muEta", "muPhi", "elPt", "elEta", "elPhi", "jetPt", "jetEta", "jetPhi", "jetMass", "metPx", "metPy", "mcT", "mcTbar"]

    with uproot.open(f"{filename}:{treename}") as tree:
        batch_data = tree.arrays(branches, entry_start=start, entry_stop=stop, library="ak")

    N_batch = len(batch_data["metPx"])
    inputs, targets, lkr_list = [], [], []
    skipped = 0

    Nmu, Nel, Njet = np.array(batch_data["Nmu"]).astype(int), np.array(batch_data["Nel"]).astype(int), np.array(batch_data["Njet"]).astype(int)
    met_px, met_py = np.array(batch_data["metPx"]).astype(np.float32), np.array(batch_data["metPy"]).astype(np.float32)

    for i in range(N_batch):
        t_arr_list, tb_arr_list = batch_data["mcT"][i].to_list(), batch_data["mcTbar"][i].to_list()
        if len(t_arr_list) < 4 or len(tb_arr_list) < 4: skipped += 1; continue
            
        t_arr, tb_arr = np.array(t_arr_list, dtype=np.float64), np.array(tb_arr_list, dtype=np.float64)
        t_px, t_py, t_pz, t_m = t_arr[0], t_arr[1], t_arr[2], t_arr[3]
        tb_px, tb_py, tb_pz, tb_m = tb_arr[0], tb_arr[1], tb_arr[2], tb_arr[3]
        t_e  = np.sqrt(t_px**2  + t_py**2  + t_pz**2  + t_m**2)
        tb_e = np.sqrt(tb_px**2 + tb_py**2 + tb_pz**2 + tb_m**2)

        if Nmu[i] < 1 or Nel[i] < 1 or Njet[i] < 2: skipped += 1; continue

        mu_pt_val, mu_eta_val, mu_phi_val = float(batch_data["muPt"][i][0]), float(batch_data["muEta"][i][0]), float(batch_data["muPhi"][i][0])
        el_pt_val, el_eta_val, el_phi_val = float(batch_data["elPt"][i][0]), float(batch_data["elEta"][i][0]), float(batch_data["elPhi"][i][0])
        
        lM = {"pt": mu_pt_val, "eta": mu_eta_val, "phi": mu_phi_val, "e": calc_e(mu_pt_val, mu_eta_val, MASS_MU)}
        lP = {"pt": el_pt_val, "eta": el_eta_val, "phi": el_phi_val, "e": calc_e(el_pt_val, el_eta_val, MASS_EL)}

        jets, jmasses = [], []
        for j in range(Njet[i]):
            jpt, jeta, jphi, jm = float(batch_data["jetPt"][i][j]), float(batch_data["jetEta"][i][j]), float(batch_data["jetPhi"][i][j]), float(batch_data["jetMass"][i][j])
            jets.append({"pt": jpt, "eta": jeta, "phi": jphi, "e": calc_e(jpt, jeta, abs(jm))})
            jmasses.append(jm)

        jb1, jb2 = select_best_jets_event(lM, lP, jets, jmasses)
        if jb1 is None: skipped += 1; continue
        def cart(v): return [v["e"], v["pt"]*np.cos(v["phi"]), v["pt"]*np.sin(v["phi"]), v["pt"]*np.sinh(v["eta"])]

        m_lpj1 = invariant_mass_2body(lP["pt"],lP["eta"],lP["phi"],lP["e"], jb1["pt"],jb1["eta"],jb1["phi"],jb1["e"])
        m_lmj2 = invariant_mass_2body(lM["pt"],lM["eta"],lM["phi"],lM["e"], jb2["pt"],jb2["eta"],jb2["phi"],jb2["e"])
        ht = lM["pt"] + lP["pt"] + jb1["pt"] + jb2["pt"] + np.sqrt(met_px[i]**2 + met_py[i]**2)

        llbar_px, llbar_py, llbar_pz, llbar_e = lM["pt"]*np.cos(lM["phi"]) + lP["pt"]*np.cos(lP["phi"]), lM["pt"]*np.sin(lM["phi"]) + lP["pt"]*np.sin(lP["phi"]), lM["pt"]*np.sinh(lM["eta"]) + lP["pt"]*np.sinh(lP["eta"]), lM["e"] + lP["e"]
        llbar_m  = np.sqrt(max(llbar_e**2 - llbar_px**2 - llbar_py**2 - llbar_pz**2, 0.0))
        ep, em = llbar_e + llbar_pz, llbar_e - llbar_pz
        llbar_rap = 0.5*np.log(ep/em) if (ep > 0 and em > 0) else 0.0

        met_pt_val  = np.sqrt(float(met_px[i])**2 + float(met_py[i])**2)
        mt_nunu     = np.sqrt(llbar_m**2 + met_pt_val**2)
        pz_nunu_lkr = mt_nunu * np.sinh(llbar_rap)
        e_nunu_lkr  = mt_nunu * np.cosh(llbar_rap)

        llnn_px, llnn_py, llnn_pz, llnn_e = llbar_px + float(met_px[i]), llbar_py + float(met_py[i]), llbar_pz + pz_nunu_lkr, llbar_e  + e_nunu_lkr
        llnn_m  = np.sqrt(max(llnn_e**2 - llnn_px**2 - llnn_py**2 - llnn_pz**2, 0.0))

        mw = 80.4
        if llnn_m < 2.0 * mw:
            llnn_pt2 = llnn_px**2 + llnn_py**2
            llnn_rap_val = 0.5*np.log((llnn_e+llnn_pz)/(llnn_e-llnn_pz+1e-10)) if (llnn_e+llnn_pz > 0 and llnn_e-llnn_pz > 0) else 0.0
            llnn_e_corr  = np.sqrt(4*mw*mw + llnn_pt2) * np.cosh(llnn_rap_val)
            llnn_pz_corr = llnn_e_corr * np.tanh(llnn_rap_val)
        else: llnn_e_corr, llnn_pz_corr  = llnn_e, llnn_pz

        def j_cart(v): return (v["pt"]*np.cos(v["phi"]), v["pt"]*np.sin(v["phi"]), v["pt"]*np.sinh(v["eta"]), v["e"])
        b1, b2 = j_cart(jb1), j_cart(jb2)
        tt_px_lkr, tt_py_lkr, tt_pz_lkr, tt_e_lkr = llnn_px + b1[0] + b2[0], llnn_py + b1[1] + b2[1], llnn_pz_corr + b1[2] + b2[2], llnn_e_corr  + b1[3] + b2[3]

        mtt_lkr   = np.sqrt(max(tt_e_lkr**2 - tt_px_lkr**2 - tt_py_lkr**2 - tt_pz_lkr**2, 0.0))
        pttt_lkr  = np.sqrt(tt_px_lkr**2 + tt_py_lkr**2)
        phitt_lkr = np.arctan2(tt_py_lkr, tt_px_lkr)
        ep_lkr, em_lkr = tt_e_lkr + tt_pz_lkr, tt_e_lkr - tt_pz_lkr
        ytt_lkr   = 0.5*np.log(ep_lkr/em_lkr) if (ep_lkr > 0 and em_lkr > 0) else 0.0

        if mtt_lkr < 300.0: skipped += 1; continue

        lkr_vars = np.array([np.log(mtt_lkr), np.log(pttt_lkr + 1.0), ytt_lkr, np.sin(phitt_lkr), np.cos(phitt_lkr)], dtype=np.float32)
        tgt = ttbar_vars(t_e, t_px, t_py, t_pz, tb_e, tb_px, tb_py, tb_pz, lkr_vars)
        if not np.all(np.isfinite(tgt)): skipped += 1; continue

        # РІВНО 26 ОЗНАК
        feat = np.array(cart(lM) + cart(lP) + cart(jb1) + cart(jb2) +
                [float(met_px[i]), float(met_py[i]), m_lpj1, m_lmj2, ht,
                 llbar_m, llbar_rap, mt_nunu, pz_nunu_lkr, llnn_m], dtype=np.float32)

        if not np.all(np.isfinite(feat)): skipped += 1; continue
        inputs.append(feat); targets.append(tgt); lkr_list.append(np.array([mtt_lkr, pttt_lkr, ytt_lkr, phitt_lkr], dtype=np.float32))

    return inputs, targets, lkr_list, skipped

def load_and_prepare(filename, treename="tree"):
    print(f"[I] Відкриваємо {filename}:{treename}")
    with uproot.open(f"{filename}:{treename}") as tree: total_events = tree.num_entries
    
    step_size = 50000
    ranges = [(filename, treename, i, min(i + step_size, total_events)) for i in range(0, total_events, step_size)]

    all_inputs, all_targets, all_lkr, total_skipped = [], [], [], 0
    max_workers = min(6, os.cpu_count() or 1)

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for idx, (inputs, targets, lkr_list, skipped) in enumerate(executor.map(process_single_batch, ranges)):
            all_inputs.extend(inputs); all_targets.extend(targets); all_lkr.extend(lkr_list); total_skipped += skipped
            print(f"[I] Батч {idx+1}/{len(ranges)} оброблено.")

    X, y, lkr = np.stack(all_inputs, axis=0), np.stack(all_targets, axis=0), np.stack(all_lkr, axis=0)
    print(f"\n[I] Готово для тренування: {len(X)} подій")
    return X, y, lkr

class SolveDataset(Dataset):
    def __init__(self, X, y, norm=None):
        if norm is None:
            norm = {
                "x_mean": X.mean(axis=0).tolist(),
                "x_std":  (X.std(axis=0) + 1e-8).tolist()
            }
        self.norm   = norm
        self.xm = np.array(norm["x_mean"], dtype=np.float32)
        self.xs = np.array(norm["x_std"],  dtype=np.float32)
        self.X_raw = X.astype(np.float32)
        self.y_raw = y.astype(np.float32)

    def __len__(self): return len(self.X_raw)

    def __getitem__(self, i):
        # АУГМЕНТАЦІЮ ПРИБРАНО ЗВІДСИ, ПЕРЕНЕСЕНО НА ВІДЕОКАРТУ
        xi = (self.X_raw[i] - self.xm) / self.xs
        yi = self.y_raw[i]
        return torch.from_numpy(xi), torch.from_numpy(yi)

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim), nn.SiLU(), nn.Linear(dim, dim), nn.LayerNorm(dim))
        self.act = nn.SiLU()
    def forward(self, x): return self.act(x + self.net(x))

class SolveMLP(nn.Module):
    def __init__(self, in_dim=IN_DIM, out_dim=OUT_DIM):
        super().__init__()
        dim = 128
        self.input_proj = nn.Sequential(nn.Linear(in_dim, dim), nn.LayerNorm(dim), nn.SiLU())
        # ЗАЛИШАЄМО 3 БЛОКИ, ЩОБ НЕ ЗЛАМАТИ weights.py ТА LKRnn.cxx
        self.blocks = nn.Sequential(ResBlock(dim), ResBlock(dim), ResBlock(dim)) 
        self.head = nn.Linear(dim, out_dim)
        
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        
    def forward(self, x): 
        raw = self.head(self.blocks(self.input_proj(x)))
        t = torch.tanh(raw)
        
        # Симетричні 15% для маси (щоб вистачило сили опустити події до 345 ГеВ)
        out_0 = t[:, 0] * 0.15
        out_1 = t[:, 1] * 0.20
        out_2 = t[:, 2] * 0.05  # Рапідність залишаємо 5%, бо вона ідеальна!
        
        return torch.stack([out_0, out_1, out_2], dim=1)

# --- КАСТОМНА ФУНКЦІЯ ВТРАТ З ВАГАМИ ---
class WeightedL1Loss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.register_buffer("weights", weights / weights.sum())

    def forward(self, pred, target):
        loss = torch.abs(pred - target) # L1 Loss (модуль)
        return (loss.mean(dim=0) * self.weights).sum()
# ---------------------------------------

class EarlyStopping:
    def __init__(self, patience=PATIENCE):
        self.patience, self.best_loss, self.counter, self.best_state = patience, float("inf"), 0, None
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - 1e-6:
            self.best_loss, self.counter, self.best_state = val_loss, 0, {k: v.clone() for k, v in model.state_dict().items()}
        else: self.counter += 1
        return self.counter >= self.patience

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[I] Використовуємо пристрій: {device}")
    
    os.makedirs(args.outdir, exist_ok=True)
    X, y, lkr = load_and_prepare(args.input, args.tree)

    N = len(X)
    idx = np.random.RandomState(SEED).permutation(N)
    n_tr, n_val = int(0.70*N), int(0.15*N)
    idx_tr, idx_val, idx_te = idx[:n_tr], idx[n_tr:n_tr+n_val], idx[n_tr+n_val:]

    # Більше не передаємо augment=True, бо ми робимо це на GPU
    ds_tr = SolveDataset(X[idx_tr], y[idx_tr])
    norm = ds_tr.norm
    ds_val = SolveDataset(X[idx_val], y[idx_val], norm)

    with open(os.path.join(args.outdir, "norm_stats.json"), "w") as f: json.dump(norm, f, indent=2)

    # ОПТИМІЗАЦІЯ DATALOADER: persistent_workers=True
    loader_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    loader_val = DataLoader(ds_val, batch_size=BATCH_SIZE*4, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    model = SolveMLP().to(device)
    
    # МАГІЯ PYTORCH 2.0 (TORCH.COMPILE)
    try:
        print("[I] Спроба компіляції моделі для максимальної швидкості (torch.compile)...")
        model = torch.compile(model)
        print("[I] Модель успішно скомпільовано!")
    except Exception as e:
        print("[W] Ваша версія або система не підтримує torch.compile. Тренування продовжиться без нього.")

    stopper = EarlyStopping(PATIENCE)
    
    # --- ВИКОРИСТАННЯ WEIGHTED L1 LOSS ---
    # Вага 5.0 для маси, 1.0 для pT і y (маса в 5 разів важливіша)
    LOSS_WEIGHTS = torch.tensor([5.0, 1.0, 1.0], device=device)
    criterion = WeightedL1Loss(LOSS_WEIGHTS)
    # -------------------------------------
    
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs, eta_min=LR*1e-2)

    print("\n[I] Починаємо тренування. Спостерігайте за GPU!")
    start_time = time.time()
    
    for ep in range(1, args.epochs + 1):
        model.train(); tr_loss = 0.0
        for xb, yb in loader_tr:
            xb, yb = xb.to(device), yb.to(device)
            
            # --- ВЕКТОРИЗОВАНА АУГМЕНТАЦІЯ НА GPU (Блискавично швидко) ---
            alpha = torch.empty(xb.shape[0], device=device).uniform_(0, 2*np.pi)
            c, s  = torch.cos(alpha), torch.sin(alpha)
            
            # Індекси Px та Py: lepM(1,2), lepP(5,6), jb1(9,10), jb2(13,14), MET(16,17)
            for base in [1, 5, 9, 13, 16]:
                px = xb[:, base].clone()
                py = xb[:, base + 1].clone()
                xb[:, base]     = px * c - py * s
                xb[:, base + 1] = px * s + py * c
            # -------------------------------------------------------------
            
            optim.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            tr_loss += loss.item() * len(xb)

        model.eval(); val_loss = 0.0
        with torch.no_grad():
            for xb, yb in loader_val: 
                val_loss += criterion(model(xb.to(device)), yb.to(device)).item() * len(xb)

        tr_loss, val_loss = tr_loss / len(ds_tr), val_loss / len(ds_val)
        scheduler.step()
        
        if ep % 10 == 0 or ep == 1:
            elapsed = time.time() - start_time
            print(f"Епоха {ep:4d} | Tr Loss: {tr_loss:8.6f} | Val Loss: {val_loss:8.6f} | Час: {elapsed:.1f}c")
            
        if stopper(val_loss, model): 
            print(f"-> Зупинка на епосі {ep} (Patience: {PATIENCE})")
            break

    # Якщо torch.compile був увімкнений, треба дістати "чисті" ваги з _orig_mod
    clean_state_dict = stopper.best_state
    if any(k.startswith('_orig_mod.') for k in clean_state_dict.keys()):
        clean_state_dict = {k.replace('_orig_mod.', ''): v for k, v in clean_state_dict.items()}

    model.load_state_dict(clean_state_dict, strict=False)
    torch.save(clean_state_dict, os.path.join(args.outdir, "solve_nn_best.pt"))
    print("\n[I] Тренування успішно завершено!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="mc_signal_merged.root")
    parser.add_argument("--tree",   default="tree")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--outdir", default="solve_nn_output")
    train(parser.parse_args())