#!/bin/bash
# Відтворює найкращі дослідницькі моделі реконструкції M, pT, y системи tt̄:
#   MLP  — пряме передбачення, лос L1 (медіана);
#   Flow — умовний normalizing flow (neural spline flow, zuko), точкова оцінка — середнє розподілу.
# Вхід обох (--inputs raw_csv_mlj, 24 числа): 4-вектори l⁻, l⁺ і двох jets, вибраних LKRv3, MET (px, py),
# дискримінанти CSV цих двох jets і чотири маси m(l⁺j₁), m(l⁺j₂), m(l⁻j₁), m(l⁻j₂).
# Аугментація: азимутальний поворот і відображення z -> −z. Поділ train/val/test — той самий, що в LKRnn.
#
# Використання (середовище з torch і zuko, напр. `conda activate hep`):
#   ./train_best.sh                      # MLP і Flow, сід 42
#   ./train_best.sh mlp                  # лише MLP
#   ./train_best.sh flow                 # лише Flow
#   SEEDS="42 1 2 3 4" ./train_best.sh   # кілька сідів (розкид між тренуваннями)
#   OUTDIR=studies/models/best ./train_best.sh      # куди писати моделі (за замовчуванням саме сюди)
#   EPOCHS=1 OUTDIR=/tmp/test ./train_best.sh       # швидка перевірка, що все запускається
#
# Потрібно: ttbar_output_full_{1,2,3}.root (вихід eventReco з flagStoreAllVars 1) і ntuples-mc/ —
# з них одноразово будується файл-супутник з CSV (ttbar_output_extra_{ch}.root, nn_extra.py).
# Час на GPU: MLP ~2–4 хв, Flow ~15–20 хв на один сід.

which=${1:-both}
SEEDS=${SEEDS:-42}
OUTDIR=${OUTDIR:-studies/models/best}
EPOCHS=${EPOCHS:-400}                    # EPOCHS=1 — швидка перевірка, що все запускається
DATA="ttbar_output_full_1.root ttbar_output_full_2.root ttbar_output_full_3.root"
COMMON="--inputs raw_csv_mlj --target direct --augment-z --dim 128 --blocks 3 --lr 5e-4 --wd 1e-2 \
        --batch 4096 --epochs $EPOCHS --patience 40 --data $DATA"
start=$(date +%s)

for f in $DATA; do
  [ -f "$f" ] || { echo "[E] немає $f — спершу ./run_full_parallel.sh config_full_output.txt"; exit 1; }
done

# 0. Файл-супутник із CSV (одноразово; перевіряє вирівнювання подій із деревами eventReco)
if [ ! -f ttbar_output_extra_1.root ] || [ ! -f ttbar_output_extra_2.root ] || [ ! -f ttbar_output_extra_3.root ]; then
  echo "=== супутник nn_extra (CSV jets) ==="
  python3 nn_extra.py --ref 'ttbar_output_full_{ch}.root' || { echo "[E] nn_extra.py впав"; exit 1; }
fi

suffix() { [ "$1" = 42 ] && echo "" || echo "_s$1"; }

for s in $SEEDS; do
  if [ "$which" = both ] || [ "$which" = mlp ]; then
    echo "=== MLP, сід $s ==="
    python3 train_study.py --model mlp --loss l1 $COMMON --seed "$s" \
        --outdir "$OUTDIR/mlp$(suffix "$s")" || { echo "[E] MLP (сід $s) впав"; exit 1; }
  fi
  if [ "$which" = both ] || [ "$which" = flow ]; then
    echo "=== Flow, сід $s ==="
    python3 train_study.py --model flow $COMMON --ctx 64 --transforms 4 --flow-hidden 128 --bins 8 \
        --samples 64 --estimator mean --seed "$s" \
        --outdir "$OUTDIR/flow$(suffix "$s")" || { echo "[E] Flow (сід $s) впав"; exit 1; }
  fi
done

echo "[I] готово за $(( $(date +%s) - start )) с; моделі — у $OUTDIR/"
echo "[I] застосувати до виходу eventReco (див. README-kinreco.md, розділ nn_apply):"
echo "    python3 nn_apply.py --input-pattern 'ttbar_output_det_{ch}.root' --output-pattern 'ttbar_output_best_{ch}.root' \\"
echo "        --det-model keep --gen-model keep --study $OUTDIR/mlp:lkrnnmlp --study $OUTDIR/flow:lkrnnflow"
