#!/bin/bash
# Тренує ПІДРЯД обидві моделі й експортує ваги:
#   1) детекторна  (--level det) -> solve_nn_output      -> kinreco/nn_weights.h
#   2) генераторна (--level gen) -> solve_nn_gen_output  -> kinreco/nn_weights_gen.h
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

echo "=========================================================="
echo " Експорт ваг у C++ header"
echo "=========================================================="
python3 weights.py        # solve_nn_output     -> kinreco/nn_weights.h      (NNWeights)
python3 weights.py --gen  # solve_nn_gen_output -> kinreco/nn_weights_gen.h  (NNWeights_gen)

end=$(date +%s.%N)
printf "[I] Обидві моделі натреновано й експортовано. Зайняло %.0f с\n" "$(echo "$end - $start" | bc)"
