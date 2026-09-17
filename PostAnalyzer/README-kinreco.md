# Команди: LKR / LKRv2 / LKRv3 / LKRnn


## Збірка

python weights.py; python weights.py --gen     # моделі .pt -> kinreco/nn_weights*.h (у git їх немає)
./compile.sh
```

## Прогін eventReco

```fish
./ttbarMakeHist config_test.txt    

#Паралельно
./run_full_parallel.sh config_full_output.txt --nngen 0     
for ch in 1 2 3
    mv ttbar_output_full_$ch.root ttbar_output_det_$ch.root  
end

./test_gennn_det.sh                                          # крос-тест gen-моделі на det -> 
```

## Аналіз

```fish
python3 integral_res.py > results_det.txt                    # bias, роздільна здатність, ефективність
python3 paper_figs.py --summary --outdir figs                # рисунки
python3 integral_res.py --level gen                          # генераторний рівень
python3 integral_res.py --input-pattern 'ttbar_output_gennn_{ch}.root' --split-model gen   # крос-тест
```

## Тренування мережі

```fish
python train_solve_nn.py --level det --inputs ttbar_output_full_{1,2,3}.root   # -> solve_nn_output/
python train_solve_nn.py --level gen --inputs ttbar_output_full_{1,2,3}.root   # -> solve_nn_gen_output/
python weights.py;
python weights.py --gen; 
./compile.sh                     
python nn_apply.py --check 'ttbar_output_det_{ch}.root'                       # звірити ваги без перекомпіляції
```

---

- `--nngen 0` — детекторна модель (основні результати), `--nngen 1` — генераторна на детекторних даних;
  у конфізі це `krNNGen`, прапорець його перекриває.
- Алгоритми оцінюються на всьому датасеті, LKRnn — на подіях поза тренуванням (`nn_split.py`);
  ефективність — на всіх подіях.
- `nn_weights*.h` не комітити; після зміни моделі — `weights.py` + `./compile.sh`.
- Дослідницькі моделі й прогони — у `studies/` (`studies/README.md`), гілка з TMVA SOFIE — `lkrnn_sofie`.
