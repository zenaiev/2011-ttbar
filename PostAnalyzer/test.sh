#!/bin/bash
# Кінематична реконструкція + метрики.
# Тип входів (детектор/генератор) задається в config_test.txt: krGenInputs.
start=$(date +%s.%N)

gen=$(awk '/^krGenInputs/{print $2}' config_test.txt)
[ "${gen:-0}" = 1 ] && tag=mc || tag=mcdet

rm -rf kr_performance/*
./ttbarMakeHist config_test.txt
cp ttbar_output_3.root "ttbar_output_${tag}.root"

python3 kinreco_eff_v3.py ttbar_output_3.root
python3 correlation.py --root ttbar_output_3.root
python3 integral_res.py ttbar_output_3.root

# порівняння gen vs детектор, коли є обидва прогони
if [ -f ttbar_output_mc.root ] && [ -f ttbar_output_mcdet.root ]; then
  python3 compare_mc.py
fi

printf "It took %.0f seconds\n" "$(echo "$(date +%s.%N) - $start" | bc)"
