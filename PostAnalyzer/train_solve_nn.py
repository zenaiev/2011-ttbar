import argparse, json, os, time
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

def load_and_prepare(inputs, treename="ttbarTree", level="det"):
    """Читає вихід eventReco: 26 готових ознак (nn_features) + база LKRv3 + gen-ціль,
    з одного чи кількох файлів (усі канали). Ознаки й база рахуються C++ (LKRv3::computeFeatures),
    тож train/inference повністю збігаються.
      level='det' -> детекторні гілки, лише reco_passed_selection==1 (вимога: події,
                     що не пройшли eventReco-відбір, у тренування НЕ потрапляють);
      level='gen' -> генераторні гілки (_gen), усі події.
    """
    if isinstance(inputs, str):
        inputs = [inputs]
    suffix = "" if level == "det" else "_gen"
    fbr = "nn_features" if level == "det" else "nn_features_gen"
    Xs, ys, lkrs, origins = [], [], [], []   # origins: (канал, ROOT-entry) кожної події
    import re
    for fname in inputs:
        print(f"[I] Відкриваємо {fname}:{treename} (рівень {level})")
        m_ch = re.search(r"_(\d+)\.root$", fname)
        channel = int(m_ch.group(1)) if m_ch else 0   # канал з імені файлу
        tree = uproot.open(f"{fname}:{treename}")
        keys = [fbr, f"mtt_lkrv3{suffix}", f"pttt_lkrv3{suffix}", f"ytt_lkrv3{suffix}",
                "mtt_gen", "pttt_gen", "ytt_gen", f"lkrv3{suffix}"]
        if level == "det":
            keys.append("reco_passed_selection")
        a = tree.arrays(keys, library="np")
        feats    = np.stack(a[fbr]).astype(np.float32)   # (N, 26) готові ознаки з C++
        mtt_lkr  = a[f"mtt_lkrv3{suffix}"]
        pttt_lkr = a[f"pttt_lkrv3{suffix}"]
        ytt_lkr  = a[f"ytt_lkrv3{suffix}"]
        mtt_gen, pttt_gen, ytt_gen = a["mtt_gen"], a["pttt_gen"], a["ytt_gen"]

        # відбір: eventReco-селекція (det), заповнені ознаки, успішний LKRv3, фізична база
        ok = (feats[:, 0] != -999) & (a[f"lkrv3{suffix}"] == 1)
        if level == "det":
            ok &= (a["reco_passed_selection"] == 1)      # <-- лише події, що пройшли eventReco
        ok &= np.isfinite(mtt_lkr) & (mtt_lkr > 300) & (mtt_lkr < 7000)
        ok &= np.isfinite(mtt_gen) & (mtt_gen > 0)

        # ціль: залишок у лог-просторі (точно як на inference LKRnn)
        log_mtt_base = np.log(np.maximum(mtt_lkr[ok], 300.0))
        log_ptt_base = np.log(pttt_lkr[ok] + 1.0)
        d_log_mtt = np.log(mtt_gen[ok]) - log_mtt_base
        d_log_ptt = np.log(pttt_gen[ok] + 1.0) - log_ptt_base
        d_ytt     = ytt_gen[ok] - ytt_lkr[ok]

        Xs.append(feats[ok])
        ys.append(np.stack([d_log_mtt, d_log_ptt, d_ytt], axis=1).astype(np.float32))
        lkrs.append(np.stack([mtt_lkr[ok], pttt_lkr[ok], ytt_lkr[ok]], axis=1).astype(np.float32))
        entries = np.nonzero(ok)[0]                      # ROOT-entry кожної відібраної події
        origins.append(np.stack([np.full(len(entries), channel), entries], axis=1))
        print(f"    відібрано {int(ok.sum())} / {len(feats)} подій")

    X = np.concatenate(Xs); y = np.concatenate(ys); lkr = np.concatenate(lkrs)
    origin = np.concatenate(origins)                     # (N, 2): канал, entry
    good = np.all(np.isfinite(y), axis=1)                # прибрати нефінітні цілі
    X, y, lkr, origin = X[good], y[good], lkr[good], origin[good]
    print(f"\n[I] Готово для тренування: {len(X)} подій ({level}-рівень, усі канали)")
    return X, y, lkr, origin

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
        
        # ЖОРСТКИЙ СИМЕТРИЧНИЙ МІКРО-ПОВІДОК (5%)
        out_0 = t[:, 0] * 0.05  # <--- ТУТ ТІЛЬКИ 0.05
        out_1 = t[:, 1] * 0.20
        out_2 = t[:, 2] * 0.05  
        
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
    # min_delta — ВІДНОСНИЙ поріг значущого покращення (0.1%): лічильник patience
    # скидається лише коли val_loss падає більш ніж на min_delta від best (щоб мікро-
    # покращення ~1e-6 не тримали тренування нескінченно). Найкращий стан зберігається
    # на БУДЬ-якому покращенні.
    def __init__(self, patience=PATIENCE, min_delta=1e-3, save_path=None):
        self.patience, self.min_delta, self.save_path = patience, min_delta, save_path
        self.best_loss, self.counter, self.best_state = float("inf"), 0, None
    @staticmethod
    def _clean(state):
        # прибрати префікс torch.compile (_orig_mod.), щоб ваги були придатні для inference
        if any(k.startswith("_orig_mod.") for k in state):
            return {k.replace("_orig_mod.", ""): v for k, v in state.items()}
        return state
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss * (1.0 - self.min_delta):
            self.counter = 0          # значуще покращення -> скидаємо
        else:
            self.counter += 1
        if val_loss < self.best_loss:  # будь-яке покращення -> оновлюємо найкращий стан
            self.best_loss = val_loss
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
            if self.save_path:        # ...і одразу пишемо на диск (Ctrl+C не втратить прогрес)
                torch.save(self._clean(self.best_state), self.save_path)
        return self.counter >= self.patience

def save_trained_root(path, origin, idx_trained, level):
    """Зберігає у ROOT перелік подій, що БРАЛИ УЧАСТЬ у навчанні NN (train+val).
    Дерево 'nn_trained' з гілками: channel, entry (позиція події у ttbarTree відповідного
    ttbar_output_full_<channel>.root) та level ('det'/'gen').
    Події, яких тут НЕМА, вважаються придатними для незалежної оцінки мережі."""
    ch = origin[idx_trained, 0].astype(np.int32)
    en = origin[idx_trained, 1].astype(np.int64)
    order = np.lexsort((en, ch))          # впорядкувати за (канал, entry) для зручності
    with uproot.recreate(path) as f:
        f["nn_trained"] = {"channel": ch[order], "entry": en[order]}
    per_ch = {int(c): int((ch == c).sum()) for c in np.unique(ch)}
    print(f"[I] Позначено {len(ch)} тренувальних подій ({level}) -> {path}")
    print(f"    по каналах: {per_ch}")


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[I] Використовуємо пристрій: {device}")
    
    os.makedirs(args.outdir, exist_ok=True)
    X, y, lkr, origin = load_and_prepare(args.inputs, args.tree, args.level)

    N = len(X)
    idx = np.random.RandomState(SEED).permutation(N)
    n_tr, n_val = int(0.70*N), int(0.15*N)
    idx_tr, idx_val, idx_te = idx[:n_tr], idx[n_tr:n_tr+n_val], idx[n_tr+n_val:]

    # Позначаємо у ROOT саме ті події, що БРАЛИ УЧАСТЬ у навчанні (train + val).
    # Логіка "позначаємо тренувальні": подія без мітки за замовчуванням придатна для
    # оцінки, тож нова/непозначена подія ніколи не буде помилково зарахована як тестова.
    # Алгоритмічних реконструкцій (LKR/LKRv3/FKR) це не стосується — вони на всьому датасеті.
    save_trained_root(os.path.join(args.outdir, "nn_trained.root"), origin,
                      np.concatenate([idx_tr, idx_val]), args.level)

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

    stopper = EarlyStopping(PATIENCE, save_path=os.path.join(args.outdir, "solve_nn_best.pt"))
    
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

    # best-модель уже збережена stopper'ом на кожному покращенні; фінально гарантуємо запис
    clean_state_dict = EarlyStopping._clean(stopper.best_state)
    model.load_state_dict(clean_state_dict, strict=False)
    torch.save(clean_state_dict, os.path.join(args.outdir, "solve_nn_best.pt"))
    print(f"\n[I] Тренування завершено. Best val loss: {stopper.best_loss:.6f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+",
                        default=["ttbar_output_full_1.root", "ttbar_output_full_2.root", "ttbar_output_full_3.root"],
                        help="вихідні файли eventReco (усі канали за замовчуванням)")
    parser.add_argument("--tree",   default="ttbarTree")
    parser.add_argument("--level",  choices=["det", "gen"], default="det",
                        help="det -> детекторні ознаки (reco_passed_selection==1); gen -> генераторні")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--outdir", default=None,
                        help="каталог виходу (за замовч. solve_nn_output, або solve_nn_gen_output для --level gen)")
    args = parser.parse_args()
    if args.outdir is None:
        args.outdir = "solve_nn_gen_output" if args.level == "gen" else "solve_nn_output"
    train(args)