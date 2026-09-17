#!/bin/bash
# Тренує ПІДРЯД обидві моделі:
#   1) детекторна  (--level det) -> solve_nn_output
#   2) генераторна (--level gen) -> solve_nn_gen_output
# У кожній папці: solve_nn_best.pt (PyTorch), solve_nn.onnx (для C++, читається під час запуску
# через ROOT TMVA SOFIE — перекомпіляція не потрібна) і dataset_split.root (мітки train/val/test).
# Послідовно, бо обидві задіюють GPU. Джерело даних — вихід eventReco (усі канали).
start=$(date +%s.%N)

echo "=========================================================="
echo " 1/2  ДЕТЕКТОРНА модель (level=det)"
echo "=========================================================="
python3 train_solve_nn.py --level det || { echo "[E] тренування det впало"; exit 1; }

echo "=========================================================="
echo " 2/2  ГЕНЕРАТОРНА модель (level=gen)"
echo "=========================================================="
python3 train_solve_nn.py --level gen || { echo "[E] тренування gen впало"; exit 1; }

end=$(date +%s.%N)
printf "[I] Обидві моделі натреновано. Зайняло %.0f с\n" "$(echo "$end - $start" | bc)"
