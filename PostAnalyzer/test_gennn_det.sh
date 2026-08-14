#!/bin/bash
# Тест: генераторно-навчена NN на ДЕТЕКТОРНОМУ MC.
# Входи детекторні (krGenInputs 0), ваги мережі генераторні (krNNGen 1),
# тестовий набір gen-моделі (test_indices_gen.txt).
# У виході mtt_lkrnn = gen-модель на детекторних входах (out-of-distribution).
start=$(date +%s.%N)

sed 's/^krGenInputs .*/krGenInputs 0/' config_test.txt > config_gennn.txt
grep -q '^krNNGen' config_gennn.txt \
  && sed -i 's/^krNNGen .*/krNNGen 1/' config_gennn.txt \
  || echo 'krNNGen 1' >> config_gennn.txt

./ttbarMakeHist config_gennn.txt
cp ttbar_output_3.root ttbar_output_gennn_det.root

# роздільна здатність / bias / ефективність / ratio
python3 kinreco_eff_v3.py ttbar_output_3.root
# зберегти під окремими іменами (щоб не перезаписати звичайні прогони)
for p in resolution resolution_ratio bias efficiency; do
  [ -f "plots/$p.png" ] && cp "plots/$p.png" "plots/${p}_gennn_det.png"
done

python3 integral_res.py ttbar_output_3.root
rm -f config_gennn.txt

echo "[I] Графіки: plots/resolution_gennn_det.png (LKRnn = gen-модель на детекторі)"
printf "It took %.0f seconds\n" "$(echo "$(date +%s.%N) - $start" | bc)"
