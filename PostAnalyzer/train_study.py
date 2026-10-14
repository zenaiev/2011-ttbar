"""
train_study.py
──────────────
Дослідницькі моделі кінематичної реконструкції tt̄ для порівняння архітектур з LKRnn.
Робочий LKRnn (train_solve_nn.py) цей скрипт не змінює.

Етап MLP:
  --target direct    мережа передбачає саму відповідь: log M, log(pT+1), y системи tt̄ (незалежно від LKRv3);
  --target residual  поправку до розв'язку LKRv3 (як LKRnn, але без обмеження ±5%);
  --inputs raw       18 «сирих» чисел: 4-вектори двох лептонів і двох jets + MET (px, py);
  --inputs features  усі 26 ознак LKRv3::computeFeatures (ще 8 похідних, серед них наближений розв'язок LKR);
  --inputs raw_extra | features_extra
                     те саме + 10 додаткових входів із файлу-супутника nn_extra.py (CSV обох jets, кількість
                     jets і b-тегів, кількість вершин, решта jets); лише детекторний рівень;
  --inputs raw_csv   18 сирих + лише CSV двох вибраних jets (nn_extra[0:2]): жодних ознак інших jets, кінематика
                     яких гірше змодельована в MC;
  --inputs raw_mlj | raw_csv_mlj | features_mlj
                     + інваріантні маси пар лептон–jet, обчислені точно як у LKRv3::computeFeatures (конвенція
                     ROOT): m(l⁺j₁), m(l⁺j₂), m(l⁻j₁), m(l⁻j₂) — обидві гіпотези спарювання; для правильної пари
                     m(ℓb) <= sqrt(m_t² − m_W²) ≈ 153 ГеВ. Маси інваріантні до обох аугментацій.
                     raw_mlj = 18 + 4, raw_csv_mlj = 18 + 2 CSV + 4, features_mlj = 26 ознак C++ + 2 відсутні маси,
                     features_csv_mlj = 26 + 2 CSV + 2 маси (усі ознаки в стилі LKRv3, лише два jets);
  --inputs pair_extra
                     пару jets вибирає класифікатор pair_select.py, а не LKRv3: 18 сирих входів із новою парою +
                     ті самі 10 додаткових для неї + імовірність і відрив вибраної пари (супутник pairsel);
  --loss l2 | l1     L2 вчить умовне середнє (мінімізує RMS), L1 — медіану;
  --flat mtt [pttt ytt]
                     пласка вага: події зважуються обернено до щільності справжнього розподілу (log M,
                     log(pT+1), y), щоб модель менше тягнула відповідь до типових значень (менший зсув у
                     бінах справжньої маси ціною більшого розкиду); --flat-max обмежує вагу.

Етап flow (--model flow):
  умовний Neural Spline Flow (zuko) моделює весь розподіл p(log M, log(pT+1), y | подія); вхід стискається
  тим самим MLP-стволом у вектор контексту. Лос — від'ємна логарифмічна правдоподібність (NLL). Точкова
  оцінка береться з розподілу: mean, median (з вибірок) або mode (градієнтний підйом по log p від медіани); вибирається суфіксом папки
  ПАПКА@median у nn_apply.py / scan_alpha.py / jes_robustness.py. Калібрувальні криві — окремо для кожної
  оцінки (calibration_<оцінка>.json). Вибірки кожної події беруться з фіксованого набору базового шуму
  (квазівипадкова послідовність Соболя), тож оцінки детерміновані й однакові в усіх скриптах.

Регуляризація:
  --augment-z   відображення z -> −z для половини подій батчу: pp-зіткнення симетричні вздовж пучка, тож
                подія з протилежними pz усіх об'єктів і −y така ж правдоподібна (M і pT не змінюються);
  --dropout p   (flow) dropout між стволом і контекстом; (mlp --arch plain) dropout після кожного шару.

Архітектура MLP (--arch):
  res    (за замовч.) вхідний шар + residual-блоки з LayerNorm і SiLU, як у LKRnn;
  plain  звичайний повнозв'язний MLP, як у Rübenach (CERN-THESIS-2023-066, model_tt.py): --blocks шарів по
         --dim вузлів, ReLU і dropout після кожного, без нормування шарів і skip-зв'язків
         (у них: --dim 800 --blocks 3 --dropout 0.25, лос MSE, Adam lr 1e-3, батч 8192).

Поділ датасету той самий, що в LKRnn (той самий фільтр, SEED і частки), тож тестові події однакові.
--seed змінює лише ініціалізацію ваг, порядок батчів і аугментацію — поділ датасету від нього не
залежить, тож моделі з різними сідами оцінюються на тих самих подіях (розкид між тренуваннями).
Результат у <outdir> (за замовчуванням studies/models/<назва за конфігом>):
  study_model.pt      ваги, нормування входу й цілі, конфігурація;
  dataset_split.root  мітки train/val/test;
  calibration.json    калібрувальні криві — середній прогноз як функція справжнього значення, по
                      подіях train (val — для вибору α); nn_apply.py --study-cal обертає їх, і зсув у бінах справжньої
                      величини зникає (розкид при цьому не змінюється).
Застосування до файлів eventReco: python nn_apply.py --study <outdir>:<назва гілки lkrnn...>
Прогін кількох варіантів в одну папку з таблицями й рисунками: run_study.py

  python train_study.py --inputs raw --target direct --loss l2
  python train_study.py --inputs raw --target direct --loss l2 --flat mtt
"""

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn

import nn_extra
import nn_split
from train_solve_nn import SEED, ResBlock, load_and_prepare

MODEL_FILE = "study_model.pt"
CAL_FILE = "calibration.json"
CAL_BINS = 30                      # квантильні біни справжніх значень для калібрувальних кривих
N_RAW = 18                         # 4-вектори lepM, lepP, j1, j2 (16) + MET px, py (2)
N_FEAT = 26                        # усі ознаки LKRv3::computeFeatures (nn_features)
INPUTS = ["raw", "features", "raw_extra", "features_extra", "pair_extra", "raw_csv", "raw_mlj", "raw_csv_mlj",
          "features_mlj", "features_csv_mlj"]
EXTRA_COLS = {"raw_csv": [0, 1], "raw_csv_mlj": [0, 1], "features_csv_mlj": [0, 1]}   # стовпці nn_extra для входу (без ключа — усі 10)
MLJ_INPUTS = {"raw_mlj": (0, 1, 2, 3), "raw_csv_mlj": (0, 1, 2, 3),   # які маси лептон–jet додати до входу
              "features_mlj": (1, 2), "features_csv_mlj": (1, 2)}   # у 26 ознаках C++ уже є m(l⁺j₁) і m(l⁻j₂) — додаємо дві відсутні
FLAT_COL = {"mtt": 0, "pttt": 1, "ytt": 2}
FLAT_BINS = {"mtt": np.linspace(np.log(340.0), np.log(1600.0), 41),    # log M
             "pttt": np.linspace(0.0, np.log(801.0), 41),                 # log(pT + 1)
             "ytt": np.linspace(-2.5, 2.5, 41)}                          # y
ROT_PX = [1, 5, 9, 13, 16]         # індекси px (py = px + 1) для азимутальної аугментації
PZ_FLIP_RAW = [3, 7, 11, 15]       # pz l⁻, l⁺, j₁, j₂ — міняють знак при відображенні z -> −z
PZ_FLIP_FEAT = [22, 24]            # y(ll) і pz(νν) серед 8 похідних ознак — теж міняють знак


def n_base(inputs):
    """Скільки стовпців nn_features бере модель (18 сирих чи всі 26; pair_extra — 18 сирих із супутника pairsel)."""
    return N_FEAT if inputs.startswith("features") else N_RAW


def extra_kind(cfg):
    """Звідки брати супутник: nn_extra.py ("extra") чи pair_select.py ("pair")."""
    return "pair" if cfg["inputs"].startswith("pair") else "extra"


def uses_extra(cfg):
    return cfg["inputs"].endswith("_extra") or cfg["inputs"] in EXTRA_COLS


def lepton_jet_masses(feats, which=(0, 1, 2, 3)):
    """Інваріантні маси пар лептон–jet із сирих nn_features (l⁻ 0:4, l⁺ 4:8, j₁ 8:12, j₂ 12:16), у порядку
    m(l⁺j₁), m(l⁺j₂), m(l⁻j₁), m(l⁻j₂). Конвенція ROOT, як у LKRv3::computeFeatures: для m² < 0 маса від'ємна
    (так буває через мітку b-тегу від'ємною масою jet). Стовпці 0 і 3 збігаються з ознаками C++ [18] і [19]."""
    f = feats.astype(np.float64)                         # як у C++: double з float-входів, назад у float32
    out = []
    for k, (lep, jet) in enumerate(((4, 8), (4, 12), (0, 8), (0, 12))):
        if k not in which:
            continue
        v = f[:, lep:lep + 4] + f[:, jet:jet + 4]
        m2 = v[:, 0] ** 2 - (v[:, 1:] ** 2).sum(1)
        out.append(np.where(m2 < 0, -np.sqrt(-np.minimum(m2, 0)), np.sqrt(np.maximum(m2, 0))))
    return np.stack(out, axis=1).astype(np.float32)


def model_inputs(feats, cfg, extra=None):
    """Вхід моделі з nn_features (N, 26) і, для *_extra, супутника (nn_extra (N, 10) або pairsel (N, 30))."""
    if uses_extra(cfg) and extra is None:
        raise SystemExit(f"[E] модель із входом {cfg['inputs']} потребує файла-супутника ({extra_kind(cfg)})")
    if extra_kind(cfg) == "pair":
        return np.ascontiguousarray(extra, dtype=np.float32)                  # усе — з супутника pairsel
    x = feats[:, :n_base(cfg["inputs"])]
    if uses_extra(cfg):
        cols = EXTRA_COLS.get(cfg["inputs"])
        x = np.concatenate([x, extra if cols is None else extra[:, cols]], axis=1)
    if cfg["inputs"] in MLJ_INPUTS:                      # маси інваріантні до поворотів і z -> −z
        x = np.concatenate([x, lepton_jet_masses(feats, MLJ_INPUTS[cfg["inputs"]])], axis=1)
    if cfg.get("jet_order") == "pt":                     # j₁ — м'якший jet (як здебільшого після LKRv3 на det)
        hard1 = np.hypot(feats[:, 9], feats[:, 10]) > np.hypot(feats[:, 13], feats[:, 14])
        x = np.array(x, dtype=np.float32)                # копія: x може бути зрізом feats
        x[hard1] = x[hard1][:, swap_permutation(cfg["inputs"], x.shape[1])]
    return np.ascontiguousarray(x, dtype=np.float32)


def extra_for_origin(files, origin, kind="extra"):
    """Супутник для подій, заданих (канал, запис) — як повертає load_and_prepare."""
    out = None
    for f in ([files] if isinstance(files, str) else files):
        ch = nn_extra.channel_of(f)
        m = origin[:, 0] == ch
        arr = nn_extra.load(f, ch, kind)
        if out is None:
            out = np.full((len(origin), arr.shape[1]), np.nan, np.float32)
        out[m] = arr[origin[m, 1]]
    if np.isnan(out).any() or (out[:, 0] == -999).any():
        raise SystemExit("[E] nn_extra: не для всіх подій є значення (інший набір файлів?)")
    return out


def augment_columns(inputs):
    """Індекси px (для повороту) і pz (для z -> −z) у векторі входу моделі."""
    nb = n_base(inputs)
    rot = list(ROT_PX)
    flip = PZ_FLIP_RAW + (PZ_FLIP_FEAT if nb == N_FEAT else [])
    if inputs.endswith("_extra"):                    # raw_csv (лише CSV) — нічого повертати й віддзеркалювати
        rot += [nb + k for k in nn_extra.ROT_PX_EXTRA]
        flip += [nb + k for k in nn_extra.PZ_FLIP_EXTRA]
    return rot, flip


def swap_permutation(inputs, x_dim):
    """Перестановка стовпців входу, що міняє місцями j₁ і j₂ (4-вектори, CSV, маси M(lj)).
    Потрібна для --augment-swap: на gen-рівні порядок jets збігається з зарядом b-кварків (j₂ — з того самого
    топа, що l⁺), на детекторі — ні; перестановка забирає в мережі цю «підказку». Для features* неможлива:
    серед 26 ознак є лише M(l⁺j₁) і M(l⁻j₂), без їхніх «пар» після перестановки."""
    if inputs.startswith("features") or extra_kind({"inputs": inputs}) == "pair" or \
            (inputs.endswith("_extra") and inputs not in EXTRA_COLS):
        raise SystemExit(f"[E] --augment-swap для входу {inputs} не підтримується")
    perm = list(range(x_dim))
    perm[8:12], perm[12:16] = list(range(12, 16)), list(range(8, 12))      # j₁ <-> j₂
    if inputs in EXTRA_COLS:                                               # CSV j₁ <-> CSV j₂
        perm[N_RAW], perm[N_RAW + 1] = N_RAW + 1, N_RAW
    if inputs in MLJ_INPUTS:                                               # M(l+j1)<->M(l+j2), M(l-j1)<->M(l-j2)
        m = x_dim - len(MLJ_INPUTS[inputs])
        perm[m:m + 4] = [m + 1, m, m + 3, m + 2]
    return perm


class StudyMLP(nn.Module):
    """MLP (як у LKRnn: вхідний шар + residual-блоки) з нормуванням входу й цілі всередині моделі."""

    def __init__(self, in_dim, out_dim=3, dim=128, blocks=3, bound=None, arch="res", dropout=0.0):
        super().__init__()
        self.bound = None if bound is None else torch.tensor(bound, dtype=torch.float32)
        self.register_buffer("x_mean", torch.zeros(in_dim))
        self.register_buffer("x_std", torch.ones(in_dim))
        self.register_buffer("y_mean", torch.zeros(out_dim))
        self.register_buffer("y_std", torch.ones(out_dim))
        if arch == "plain":                            # Dense(dim, ReLU) + Dropout, blocks разів (Rübenach)
            layers = []
            for k in range(blocks):
                layers += [nn.Linear(in_dim if k == 0 else dim, dim), nn.ReLU(), nn.Dropout(dropout)]
            self.input_proj = nn.Identity()
            self.blocks = nn.Sequential(*layers)
        else:
            self.input_proj = nn.Sequential(nn.Linear(in_dim, dim), nn.LayerNorm(dim), nn.SiLU())
            self.blocks = nn.Sequential(*[ResBlock(dim) for _ in range(blocks)])
        self.head = nn.Linear(dim, out_dim)
        if bound is not None:                          # як SolveMLP у LKRnn: старт із нульової поправки
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def _raw(self, x):
        x = (x - self.x_mean) / (self.x_std + 1e-8)
        return self.head(self.blocks(self.input_proj(x)))

    def forward_std(self, x):
        """Вихід у нормованому масштабі цілі (для лосу)."""
        if self.bound is None:
            return self._raw(x)
        return (self.forward(x) - self.y_mean) / self.y_std

    def forward(self, x):
        """Ціль у фізичному масштабі: direct -> (log M, log(pT+1), y); residual -> поправки до LKRv3.
        З bound — обмежена поправка tanh(·)·bound, як у LKRnn (±5% по log M, ±20% по log(pT+1), ±0.05 по y)."""
        if self.bound is None:
            return self._raw(x) * self.y_std + self.y_mean
        return torch.tanh(self._raw(x)) * self.bound.to(x.device)


FLOW_ESTIMATORS = ("mean", "median", "mode")


class StudyFlow(nn.Module):
    """Умовний normalizing flow (Neural Spline Flow, zuko) для p(ціль | подія).
    Нормування входу й цілі — буфери, як у StudyMLP; MLP-ствол (input_proj + ResBlocks) дає контекст для flow.
    forward(x) повертає точкову оцінку self.estimator у фізичному масштабі цілі — як StudyMLP.forward."""

    def __init__(self, in_dim, out_dim=3, dim=128, blocks=3, ctx=64, transforms=4, hidden=128, bins=8, samples=64,
                 dropout=0.0):
        super().__init__()
        import zuko
        self.register_buffer("x_mean", torch.zeros(in_dim))
        self.register_buffer("x_std", torch.ones(in_dim))
        self.register_buffer("y_mean", torch.zeros(out_dim))
        self.register_buffer("y_std", torch.ones(out_dim))
        self.input_proj = nn.Sequential(nn.Linear(in_dim, dim), nn.LayerNorm(dim), nn.SiLU())
        self.blocks = nn.Sequential(*[ResBlock(dim) for _ in range(blocks)])
        self.drop = nn.Dropout(dropout)                  # параметрів не має: старі моделі завантажуються як раніше
        self.ctx = nn.Linear(dim, ctx)
        self.flow = zuko.flows.NSF(features=out_dim, context=ctx, transforms=transforms,
                                   hidden_features=(hidden, hidden), bins=bins)
        self.estimator = "median"
        self.set_samples(samples)

    def set_samples(self, n):
        """Фіксований базовий шум N(0, 1): скрембльована послідовність Соболя, n точок (детерміновано)."""
        sob = torch.quasirandom.SobolEngine(self.y_mean.numel(), scramble=True, seed=1234).draw(n).double()
        z = torch.erfinv(2 * sob.clamp(1e-6, 1 - 1e-6) - 1) * np.sqrt(2)
        self.register_buffer("z_fixed", z.float().to(self.y_mean.device), persistent=False)

    def context(self, x):
        x = (x - self.x_mean) / (self.x_std + 1e-8)
        return self.ctx(self.drop(self.blocks(self.input_proj(x))))

    def nll_std(self, x, y_std):
        """−log p(нормована ціль | x) для кожної події (для лосу)."""
        return -self.flow(self.context(x)).log_prob(y_std)

    ROWS = 131072            # рядків «подія × вибірка» за раз (~0.6 ГБ GPU); швидкість від цього не залежить
    ASCENT_STEPS, ASCENT_LR = 30, 0.05

    def _sample_estimate(self, x, kind):
        """Середнє чи медіана з фіксованих вибірок (нормований масштаб цілі)."""
        out, n_s = [], len(self.z_fixed)
        step = max(1, self.ROWS // n_s)
        for i in range(0, len(x), step):
            dist = self.flow(self.context(x[i:i + step]))
            n = min(step, len(x) - i)
            s = dist.transform.inv(self.z_fixed[:, None, :].expand(n_s, n, -1))
            out.append(s.mean(0) if kind == "mean" else s.median(0).values)
            del dist, s
            gc.collect()        # об'єкти розподілів zuko/torch мають цикли посилань: без цього пам'ять GPU накопичується
        return torch.cat(out)

    def _mode(self, x, y0):
        """Мода: градієнтний підйом по log p від y0 (медіани); лишається краща з двох точок за log p.
        Градієнт береться лише за y — параметри моделі не зачіпаються."""
        out = []
        for i in range(0, len(x), self.ROWS):
            with torch.no_grad():
                c = self.context(x[i:i + self.ROWS])
            start = y0[i:i + self.ROWS]
            y = start.clone().requires_grad_(True)
            opt = torch.optim.Adam([y], lr=self.ASCENT_LR)
            for _ in range(self.ASCENT_STEPS):
                (g,) = torch.autograd.grad(-self.flow(c).log_prob(y).sum(), y)
                y.grad = g
                opt.step()
            with torch.no_grad():
                dist = self.flow(c)
                better = dist.log_prob(y) > dist.log_prob(start)
                out.append(torch.where(better[:, None], y.detach(), start))
            del dist, y, opt
            gc.collect()
        return torch.cat(out)

    def forward(self, x):
        with torch.no_grad():
            est = self._sample_estimate(x, "mean" if self.estimator == "mean" else "median")
        if self.estimator == "mode":
            with torch.enable_grad():
                est = self._mode(x, est)
        return est * self.y_std + self.y_mean


def split_spec(model_dir):
    """'ПАПКА' або 'ПАПКА@оцінка' (для flow: mean | median | mode) -> (папка, оцінка або None)."""
    d, _, est = model_dir.partition("@")
    return d, (est or None)


def make_targets(y_res, lkr, target):
    """y_res — поправки до LKRv3 з load_and_prepare; для direct додаємо назад базу LKRv3.
    lkrv3 — сам розв'язок LKRv3 замість правди (перевірка, чи мережа відтворює аналітичний алгоритм)."""
    if target == "residual":
        return y_res.astype(np.float32)
    base = np.stack([np.log(lkr[:, 0]), np.log(lkr[:, 1] + 1.0), lkr[:, 2]], axis=1)
    if target == "lkrv3":
        return base.astype(np.float32)
    return (y_res + base).astype(np.float32)


def flat_weights(truth, idx_tr, variables, wmax):
    """Ваги ~ 1/щільність справжнього розподілу (по тренувальній частині) для кожної змінної з variables,
    перемножені, обрізані зверху на wmax (відносно середньої щільності бінів) і нормовані на середнє 1."""
    w = np.ones(len(truth))
    for v in variables:
        edges = FLAT_BINS[v]
        b = np.clip(np.digitize(truth[:, FLAT_COL[v]], edges) - 1, 0, len(edges) - 2)
        cnt = np.bincount(b[idx_tr], minlength=len(edges) - 1).astype(float)
        dens = np.where(cnt > 0, cnt, cnt[cnt > 0].min())
        w *= (cnt[cnt > 0].mean() / dens)[b]
    clipped = float(np.mean(w[idx_tr] > wmax))
    w = np.minimum(w, wmax)
    w /= w[idx_tr].mean()
    return w.astype(np.float32), clipped


def lkr_base(mtt3, pttt3, ytt3):
    """База LKRv3 у просторі прямої цілі (log M, log(pT+1), y) — та сама, що в make_targets і study_corrections."""
    f32 = np.float32
    with np.errstate(all="ignore"):
        return np.stack([np.log(np.where(mtt3 > 0, mtt3, 1)).astype(f32),
                         np.log(np.maximum(pttt3, 0).astype(f32) + f32(1)),
                         np.where(np.isfinite(ytt3), ytt3, 0).astype(f32)], axis=1)


def calibrate(pred, cal, alpha=1.0, base=None):
    """apply_calibration з урахуванням простору кривих. Для моделей-поправок криві будуються в просторі кінцевої
    величини (база LKRv3 + поправка, space = 'direct'): стиснення шкали — це залежність прогнозованої МАСИ від
    справжньої маси, а не поправки від поправки. Результат повертається в простір виходу моделі."""
    if cal.get("space") == "direct":
        if base is None:
            raise ValueError("криві в просторі кінцевої величини: потрібна база LKRv3 (lkr_base)")
        return apply_calibration(pred + base, cal, alpha) - base
    return apply_calibration(pred, cal, alpha)


def fit_calibration(model, X, Y, rows, device, target, batch=65536, fit_on="train", base=None):
    """Калібрувальні криві: середній прогноз як функція справжнього значення (у просторі цілі),
    по подіях rows (train; val лишається для вибору α). Обернення цих кривих під час застосування прибирає зсув у бінах
    справжньої величини (numerical inversion, як у калібруванні енергії jets)."""
    preds, t0 = [], time.time()
    with torch.no_grad():
        for i in range(0, len(rows), batch):
            preds.append(model(torch.from_numpy(X[rows[i:i + batch]]).to(device)).cpu().numpy())
            done = min(i + batch, len(rows))
            if len(rows) > 4 * batch and (done // batch) % 2 == 0 or done == len(rows):
                print(f"    калібрування: {done}/{len(rows)} подій, {time.time() - t0:.0f} с", flush=True)
    preds = np.concatenate(preds)
    truth = Y[rows]
    space = "model"
    if target == "residual" and base is not None:       # поправка -> кінцева величина (база LKRv3 + поправка)
        preds, truth = preds + base[rows], truth + base[rows]
        space = "direct"
    cal = {"target": target, "space": space, "n_events": int(len(rows)), "fit_on": fit_on, "dims": []}
    for d in range(truth.shape[1]):
        t, p = truth[:, d], preds[:, d]
        edges = np.unique(np.quantile(t, np.linspace(0, 1, CAL_BINS + 1)))
        b = np.clip(np.digitize(t, edges) - 1, 0, len(edges) - 2)
        tm = np.array([t[b == i].mean() for i in range(len(edges) - 1)])
        pm = np.array([p[b == i].mean() for i in range(len(edges) - 1)])
        order = np.argsort(tm)
        tm, pm = tm[order], np.maximum.accumulate(pm[order])   # монотонність -> однозначне обернення
        cal["dims"].append({"true": tm.tolist(), "pred": pm.tolist()})
    return cal


def apply_calibration(pred, cal, alpha=1.0):
    """Обернення калібрувальних кривих: прогноз -> калібрований прогноз (у тому самому просторі цілі).
    За межами крайніх вузлів крива лінійно продовжується: без цього прогнози «прилипають» до крайніх
    вузлів і в крайніх бінах з'являється фальшиво малий розкид.
    alpha — часткове калібрування: 0 — сирий прогноз, 1 — повне обернення. Повне знімає зсув у бінах
    справжньої величини, але множить розкид на 1/нахил (L2 дає стиснуту відповідь саме тому, що вона
    мінімізує RMS), тож проміжні alpha дають компроміс."""
    out = np.asarray(pred, np.float32).copy()
    for d, c in enumerate(cal["dims"]):
        p, t = np.asarray(c["pred"], float), np.asarray(c["true"], float)
        v = np.asarray(pred[:, d], float)
        r = np.interp(v, p, t)
        if len(p) > 2:
            for m, i, j in ((v < p[0], 0, 1), (v > p[-1], -1, -2)):
                if m.any() and p[i] != p[j]:
                    r[m] = t[i] + (v[m] - p[i]) * (t[i] - t[j]) / (p[i] - p[j])
        out[:, d] = pred[:, d] + alpha * (r - pred[:, d]) if alpha != 1.0 else r
    return out


def cal_file(model_dir):
    """Шлях до калібрувальних кривих: calibration.json, для flow з оцінкою — calibration_<оцінка>.json."""
    d, est = split_spec(model_dir)
    return os.path.join(d, f"calibration_{est}.json" if est else CAL_FILE)


def load_calibration(model_dir):
    """Калібрувальні криві моделі (або None)."""
    p = cal_file(model_dir)
    return json.load(open(p)) if os.path.exists(p) else None


def auto_outdir(cfg):
    """Назва папки моделі за її налаштуваннями, напр. studies/models/mlp_raw_direct_l2_flatM."""
    name = f"{cfg['model']}_{cfg['inputs']}_{cfg['target']}_{cfg['loss']}"
    if cfg.get("flat"):
        name += "_flat" + "".join({"mtt": "M", "pttt": "P", "ytt": "Y"}[v] for v in cfg["flat"])
    if cfg.get("train_frac", 1.0) != 1.0:
        name += f"_f{cfg['train_frac']:g}"
    if cfg.get("bound"):
        name += "_bound"
    if not cfg.get("augment", True):
        name += "_noaug"
    if cfg.get("augment_z"):
        name += "_z"
    if cfg.get("augment_swap"):
        name += "_swap"
    if cfg.get("jet_order"):
        name += f"_order{cfg['jet_order']}"
    if cfg.get("arch", "res") != "res":
        name += f"_{cfg['arch']}"
    if cfg.get("dropout"):
        name += f"_do{cfg['dropout']:g}"
    if cfg.get("level", "det") == "gen":
        name += "_gen"
    if cfg.get("seed", SEED) != SEED:
        name += f"_s{cfg['seed']}"
    return os.path.join("studies", "models", name)


def load_study(model_dir, device="cpu"):
    """(модель, конфігурація) з <model_dir>/study_model.pt."""
    d, est = split_spec(model_dir)
    ck = torch.load(os.path.join(d, MODEL_FILE), map_location="cpu")
    cfg = dict(ck["config"])
    if cfg["model"] == "mlp":
        if est:
            raise RuntimeError(f"{model_dir}: суфікс @{est} має сенс лише для flow")
        m = StudyMLP(cfg["in_dim"], dim=cfg["dim"], blocks=cfg["blocks"], bound=cfg.get("bound"),
                     arch=cfg.get("arch", "res"), dropout=cfg.get("dropout", 0.0))
        m.load_state_dict(ck["state_dict"])
    elif cfg["model"] == "flow":
        m = StudyFlow(cfg["in_dim"], dim=cfg["dim"], blocks=cfg["blocks"], ctx=cfg["ctx"], transforms=cfg["transforms"],
                      hidden=cfg["flow_hidden"], bins=cfg["bins"], samples=cfg["samples"], dropout=cfg.get("dropout", 0.0))
        m.load_state_dict(ck["state_dict"])
        m.estimator = est or cfg.get("estimator", "median")
        if m.estimator not in FLOW_ESTIMATORS:
            raise RuntimeError(f"{model_dir}: невідома оцінка {m.estimator} (можна: {', '.join(FLOW_ESTIMATORS)})")
        cfg["estimator"] = m.estimator
    else:
        raise RuntimeError(f"{model_dir}: модель типу {cfg['model']} не підтримується")
    return m.to(device).eval(), cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["mlp", "flow"], default="mlp")
    ap.add_argument("--inputs", choices=INPUTS, required=True)
    ap.add_argument("--target", choices=["direct", "residual", "lkrv3"], required=True,
                    help="direct — (ln M, ln(pT+1), y) правди; residual — поправка до LKRv3; "
                         "lkrv3 — сам розв'язок LKRv3 як ціль (перевірка відтворення алгоритму)")
    ap.add_argument("--loss", choices=["l2", "l1"], default="l2")
    ap.add_argument("--loss-weights", dest="loss_weights", type=float, nargs=3, default=[1.0, 1.0, 1.0],
                    help="ваги M, pT, y у лосі (у нормованому масштабі цілі)")
    ap.add_argument("--flat", nargs="*", choices=["mtt", "pttt", "ytt"], default=[],
                    help="пласка вага за справжніми змінними (напр. --flat mtt або --flat mtt pttt ytt)")
    ap.add_argument("--flat-max", dest="flat_max", type=float, default=30.0,
                    help="верхня межа ваги відносно середньої щільності бінів")
    ap.add_argument("--level", choices=["det", "gen"], default="det",
                    help="det — детекторні входи (reco_passed_selection == 1); gen — генераторні входи, усі події")
    ap.add_argument("--data", nargs="+",
                    default=["ttbar_output_full_1.root", "ttbar_output_full_2.root", "ttbar_output_full_3.root"],
                    help="вихід eventReco для тренування (ознаки й база LKRv3 від мережі не залежать)")
    ap.add_argument("--outdir", default=None, help="за замовчуванням studies/models/<назва за конфігом>")
    ap.add_argument("--force", action="store_true", help="перезаписати модель, якщо в --outdir вона вже є")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--no-augment", dest="augment", action="store_false", help="без азимутальної аугментації")
    ap.add_argument("--augment-swap", dest="augment_swap", action="store_true",
                    help="у половини подій батчу міняти місцями j₁ і j₂ (разом із CSV і масами M(lj))")
    ap.add_argument("--jet-order", dest="jet_order", choices=["pt"], default=None,
                    help="pt — упорядкувати j₁, j₂ за pT (j₁ м'якший) на вході моделі; на gen-рівні прибирає "
                         "збіг порядку jets із зарядом b-кварків (j₁ = b̄, j₂ = b), на det це вже так у ~92%% подій")
    ap.add_argument("--augment-z", dest="augment_z", action="store_true",
                    help="відображення z -> −z для половини подій батчу (pz усіх об'єктів і знак y)")
    ap.add_argument("--dropout", type=float, default=0.0,
                    help="flow: dropout між стволом і контекстом; mlp --arch plain: після кожного шару")
    ap.add_argument("--arch", choices=["res", "plain"], default="res",
                    help="MLP: res — residual-блоки (як LKRnn); plain — Dense+ReLU+Dropout (як Rübenach)")
    ap.add_argument("--bound", type=float, nargs=3, default=None, metavar=("M", "PT", "Y"),
                    help="MLP з --target residual: обмежити поправку tanh(·)·bound, як у LKRnn (0.05 0.20 0.05)")
    ap.add_argument("--ctx", type=int, default=64, help="flow: розмір вектора контексту")
    ap.add_argument("--transforms", type=int, default=4, help="flow: кількість сплайн-перетворень")
    ap.add_argument("--flow-hidden", dest="flow_hidden", type=int, default=128, help="flow: ширина шарів кондиціонера")
    ap.add_argument("--bins", type=int, default=8, help="flow: вузлів сплайна")
    ap.add_argument("--samples", type=int, default=64,
                    help="flow: вибірок на подію для середнього/медіани (64 дають похибку медіани ~0.5%% маси, середнього ~0.05%%)")
    ap.add_argument("--estimator", choices=list(FLOW_ESTIMATORS), default="median",
                    help="flow: оцінка за замовчуванням (калібрування будується для всіх трьох)")
    ap.add_argument("--print-every", dest="print_every", type=int, default=10,
                    help="друкувати рядок кожні N епох (1 — щоепохи: видно, якщо тренування сповільнилось)")
    ap.add_argument("--train-frac", dest="train_frac", type=float, default=1.0,
                    help="тренувати лише на цій частці тренувальної частини (крива навчання); "
                         "поділ датасету й вибірки val/test не змінюються, тож числа лишаються порівнянними")
    ap.add_argument("--seed", type=int, default=SEED,
                    help=f"сід тренування (ініціалізація, батчі, аугментація); поділ датасету завжди з сідом {SEED}")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = {"model": args.model, "inputs": args.inputs, "target": args.target,
           "loss": "nll" if args.model == "flow" else args.loss,
           "loss_weights": args.loss_weights, "level": args.level, "dim": args.dim, "blocks": args.blocks,
           "lr": args.lr, "wd": args.wd, "batch": args.batch, "augment": args.augment,
           "flat": args.flat, "flat_max": args.flat_max, "seed": args.seed, "augment_z": args.augment_z,
           "augment_swap": args.augment_swap, "jet_order": args.jet_order,
           "bound": args.bound, "train_frac": args.train_frac}
    if uses_extra({"inputs": args.inputs}) and args.level != "det":
        raise SystemExit("[E] додаткові входи (nn_extra) є лише на детекторному рівні")
    if args.bound and (args.model != "mlp" or args.target != "residual"):
        raise SystemExit("[E] --bound має сенс лише для MLP з --target residual (поправка до LKRv3, як у LKRnn)")
    if args.model == "mlp" and args.arch != "res":
        cfg.update(arch=args.arch, dropout=args.dropout)
    if args.model == "flow":
        cfg.update(ctx=args.ctx, transforms=args.transforms, flow_hidden=args.flow_hidden, bins=args.bins,
                   samples=args.samples, estimator=args.estimator, dropout=args.dropout)
    args.outdir = args.outdir or auto_outdir(cfg)
    if os.path.exists(os.path.join(args.outdir, MODEL_FILE)) and not args.force:
        raise SystemExit(f"[E] у {args.outdir} вже є модель — інша --outdir або --force для перезапису")
    os.makedirs(args.outdir, exist_ok=True)
    print(f"[I] папка моделі: {args.outdir}")

    X, y_res, lkr, origin = load_and_prepare(args.data, "ttbarTree", args.level)
    X = model_inputs(X, cfg, extra_for_origin(args.data, origin, extra_kind(cfg)) if uses_extra(cfg) else None)
    Y = make_targets(y_res, lkr, args.target)
    cfg["in_dim"] = int(X.shape[1])

    # той самий поділ, що в LKRnn (train_solve_nn.train)
    N = len(X)
    idx = np.random.RandomState(SEED).permutation(N)
    n_tr, n_val = int(nn_split.SPLIT_FRACTIONS[0] * N), int(nn_split.SPLIT_FRACTIONS[1] * N)
    idx_tr, idx_val, idx_te = idx[:n_tr], idx[n_tr:n_tr + n_val], idx[n_tr + n_val:]
    idx_tr_full = idx_tr                                    # у файл міток завжди йде повний поділ
    if args.train_frac < 1.0:      # менше тренувальних подій; мітки поділу лишаються повними
        n_keep = int(args.train_frac * len(idx_tr))
        print(f"[I] крива навчання: тренуємо на {n_keep} з {len(idx_tr)} подій train ({100 * args.train_frac:.0f}%)")
        idx_tr = idx_tr[:n_keep]
    nn_split.write_split(nn_split.split_file(args.outdir), origin, idx_tr_full, idx_val, idx_te, args.level,
                         {"inputs": list(args.data), "seed": SEED, "study": cfg})

    truth = make_targets(y_res, lkr, "direct")          # log M, log(pT+1), y — справжні значення
    if args.flat:
        W, clipped = flat_weights(truth, idx_tr, args.flat, args.flat_max)
        wt = W[idx_tr]
        print(f"[I] пласка вага за {'+'.join(args.flat)}: max {wt.max():.1f}, обрізано {100 * clipped:.2f}% подій, "
              f"ефективна кількість подій {wt.sum() ** 2 / (wt ** 2).sum() / len(wt) * 100:.1f}% від {len(wt)}")
    else:
        W = np.ones(N, np.float32)

    if args.model == "flow":
        model = StudyFlow(cfg["in_dim"], dim=args.dim, blocks=args.blocks, ctx=args.ctx, transforms=args.transforms,
                          hidden=args.flow_hidden, bins=args.bins, samples=args.samples, dropout=args.dropout)
        model.estimator = args.estimator
    else:
        model = StudyMLP(cfg["in_dim"], dim=args.dim, blocks=args.blocks, bound=args.bound,
                         arch=args.arch, dropout=args.dropout if args.arch == "plain" else 0.0)
    with torch.no_grad():
        model.x_mean.copy_(torch.from_numpy(X[idx_tr].mean(0)))
        model.x_std.copy_(torch.from_numpy(X[idx_tr].std(0) + 1e-8))
        model.y_mean.copy_(torch.from_numpy(Y[idx_tr].mean(0)))
        model.y_std.copy_(torch.from_numpy(Y[idx_tr].std(0)))
    model.to(device)
    print(f"[I] модель: {args.model.upper()}{' plain' if args.arch == 'plain' else ''} dim={args.dim}, blocks={args.blocks}"
          + (f", NSF: {args.transforms} перетворень, кондиціонер {args.flow_hidden}, контекст {args.ctx}, {args.bins} вузлів"
             if args.model == "flow" else "")
          + f", вхід {args.inputs} ({cfg['in_dim']}), ціль {args.target}, "
          f"лос {cfg['loss']}; параметрів {sum(p.numel() for p in model.parameters()):,}; пристрій {device}")

    Xg, Yg = torch.from_numpy(X).to(device), torch.from_numpy(Y).to(device)
    base_log_m = torch.from_numpy(np.log(lkr[:, 0]).astype(np.float32)).to(device)
    tr = torch.from_numpy(idx_tr).to(device)
    va = torch.from_numpy(idx_val).to(device)
    w = torch.tensor(args.loss_weights, device=device)
    Wg = torch.from_numpy(W).to(device)

    def loss_fn(pred_std, y, ev_w):
        diff = pred_std - (y - model.y_mean) / model.y_std
        per = diff.pow(2) if args.loss == "l2" else diff.abs()
        per_event = (per * w).sum(1) / w.sum()
        return (per_event * ev_w).sum() / ev_w.sum()

    def batch_loss(xb, yb, ev_w):
        if args.model == "flow":
            per_event = model.nll_std(xb, (yb - model.y_mean) / model.y_std)
            return (per_event * ev_w).sum() / ev_w.sum()
        return loss_fn(model.forward_std(xb), yb, ev_w)

    def mass(pred_or_y, rows):
        """M у ГеВ з цілі чи прогнозу (для контролю σ(M) на валідації)."""
        lm = pred_or_y[:, 0] if args.target != "residual" else pred_or_y[:, 0] + base_log_m[rows]
        return torch.exp(lm)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 1e-2)
    best, best_state, counter = float("inf"), None, 0
    rot_cols, flip_cols = augment_columns(args.inputs)
    swap_perm = torch.tensor(swap_permutation(args.inputs, cfg["in_dim"]), device=device) if args.augment_swap else None
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = tr[torch.randperm(len(tr), device=device)]
        tr_sum = torch.zeros((), device=device)
        for i in range(0, len(perm), args.batch):
            b = perm[i:i + args.batch]
            xb = Xg[b].clone()
            if args.augment:
                a = torch.rand(len(b), device=device) * (2 * np.pi)
                c, s = torch.cos(a), torch.sin(a)
                for k in rot_cols:
                    px, py = xb[:, k].clone(), xb[:, k + 1].clone()
                    xb[:, k], xb[:, k + 1] = px * c - py * s, px * s + py * c
            yb = Yg[b]
            if swap_perm is not None:                         # j₁ <-> j₂ (порядок jets не несе інформації)
                sw = (torch.rand(len(b), device=device) < 0.5)[:, None]
                xb = torch.where(sw, xb[:, swap_perm], xb)
            if args.augment_z:                                # z -> −z: pz усіх об'єктів і знак y
                flip = (torch.rand(len(b), device=device) < 0.5)[:, None]
                xb[:, flip_cols] = torch.where(flip, -xb[:, flip_cols], xb[:, flip_cols])
                yb = yb.clone()
                yb[:, 2] = torch.where(flip[:, 0], -yb[:, 2], yb[:, 2])
            opt.zero_grad(set_to_none=True)
            loss = batch_loss(xb, yb, Wg[b])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_sum += loss.detach() * len(b)
        sched.step()

        model.eval()
        with torch.no_grad():
            if args.model == "flow":
                val = float(sum((model.nll_std(Xg[va[i:i + 65536]], (Yg[va[i:i + 65536]] - model.y_mean) / model.y_std)
                                 * Wg[va[i:i + 65536]]).sum() for i in range(0, len(va), 65536)) / Wg[va].sum())
                if ep == 1 or ep % args.print_every == 0:     # контроль σ(M): оцінка з 64 вибірок на 20 тис. подій val
                    sub = va[:20000]
                    model.set_samples(64)
                    dm = mass(model(Xg[sub]), sub) - mass(Yg[sub], sub)
                    model.set_samples(args.samples)
            else:
                preds = torch.cat([model(Xg[va[i:i + 65536]]) for i in range(0, len(va), 65536)])
                val = float(sum(loss_fn((preds[i:i + 65536] - model.y_mean) / model.y_std, Yg[va[i:i + 65536]],
                                        Wg[va[i:i + 65536]]) * Wg[va[i:i + 65536]].sum() for i in range(0, len(va), 65536))
                            / Wg[va].sum())
                dm = mass(preds, va) - mass(Yg[va], va)
        improved = val < best
        if val < best * (1 - 1e-4):
            counter = 0
        else:
            counter += 1
        if improved:
            best = val
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            torch.save({"state_dict": best_state, "config": cfg}, os.path.join(args.outdir, MODEL_FILE))
        if ep == 1 or ep % args.print_every == 0 or counter >= args.patience:
            print(f"Епоха {ep:4d} | train {float(tr_sum) / len(tr):.5f} | val {val:.5f} | val σ(M) {float(dm.std()):6.1f} ГеВ, "
                  f"bias(M) {float(dm.mean()):+6.1f} ГеВ | {time.time() - t0:.0f} с{' *' if improved else ''}")
        if counter >= args.patience:
            print(f"[I] зупинка на епосі {ep} (patience {args.patience})")
            break

    torch.save({"state_dict": best_state, "config": cfg}, os.path.join(args.outdir, MODEL_FILE))
    model.load_state_dict(best_state)
    model.eval()
    # конфігурація — одразу: модель повністю збережена, навіть якщо калібрування далі зірветься
    with open(os.path.join(args.outdir, "study_config.json"), "w") as f:
        json.dump({**cfg, "best_val": best, "train_seconds": round(time.time() - t0)}, f, indent=2)
    print(f"[I] найкраща val {best:.5f}; модель -> {os.path.join(args.outdir, MODEL_FILE)}")

    if args.model == "flow":
        # Калібрування flow — окремим свіжим процесом: одного разу в тренувальному процесі після ~16 хв роботи
        # запуски на GPU перестали виконуватись (GPU 0%, CPU 100%), а в новому процесі та сама модель рахувала
        # з нормальною швидкістю. Окремий процес заодно звільняє всю пам'ять тренування.
        del Xg, Yg, Wg, opt, sched, tr, va, base_log_m, model
        gc.collect()
        torch.cuda.empty_cache()
        cmd = [sys.executable, "calibrate_study.py", "--force", "--fit-on", "train", "--data", *args.data,
               "--model-dir", *[f"{args.outdir}@{e}" for e in FLOW_ESTIMATORS]]
        print("[I] калібрування окремим процесом: " + " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        shutil.copy2(cal_file(f"{args.outdir}@{args.estimator}"), os.path.join(args.outdir, CAL_FILE))
        print(f"[I] {CAL_FILE} = криві оцінки за замовчуванням ({args.estimator})")
        return

    base = make_targets(np.zeros_like(y_res), lkr, "direct") if args.target == "residual" else None
    cal = fit_calibration(model, X, Y, idx_tr, device, args.target, base=base)   # лише train: val — для вибору α
    with open(os.path.join(args.outdir, CAL_FILE), "w") as f:
        json.dump(cal, f, indent=1)
    d0 = cal["dims"][0]
    slope = (d0["pred"][-1] - d0["pred"][0]) / (d0["true"][-1] - d0["true"][0])
    print(f"[I] калібрування -> {os.path.join(args.outdir, CAL_FILE)} (нахил відповіді за першою змінною {slope:.3f}; "
          f"1.0 = без стискання)")

if __name__ == "__main__":
    main()
