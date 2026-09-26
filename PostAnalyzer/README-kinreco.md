# Команди: LKR / LKRv2 / LKRv3 / LKRnn


## Збірка

```fish
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

## Найкращі моделі: MLP і Flow

```fish
./train_best.sh                       # MLP і Flow, сід 42 -> studies/models/best/{mlp,flow}
./train_best.sh mlp                   # лише MLP (~3 хв на GPU); ./train_best.sh flow — лише Flow (~15 хв)
SEEDS="42 1 2 3 4" ./train_best.sh    # кілька сідів -> mlp, mlp_s1, ... (розкид між тренуваннями)
EPOCHS=1 OUTDIR=/tmp/t ./train_best.sh   # швидка перевірка, що все запускається
```

Обидві моделі передбачають M, pT, y системи tt̄ напряму (без бази LKRv3), вхід — 24 числа: 4-вектори
l⁻, l⁺ і двох jets, вибраних LKRv3, MET (px, py), дискримінанти CSV цих двох jets і маси m(l⁺j₁), m(l⁺j₂),
m(l⁻j₁), m(l⁻j₂). MLP — лос L1; Flow — normalizing flow, точкова оцінка — середнє розподілу. Потрібні
`ttbar_output_full_{1,2,3}.root` і `ntuples-mc/`: з них скрипт одноразово будує файл-супутник із CSV
`ttbar_output_extra_{ch}.root` (`nn_extra.py`; у самих `ttbar_output_*` значень CSV немає).

## nn_apply: застосування моделей до виходу eventReco

`nn_apply.py` рахує гілки мереж у Python, без перекомпіляції C++: бере готовий файл eventReco, копіює всі
гілки й дописує нові, результат — у новий файл (вхід не змінюється).

```fish
python nn_apply.py --input-pattern 'ttbar_output_det_{ch}.root' --output-pattern 'ttbar_output_best_{ch}.root' \
    --det-model keep --gen-model keep \
    --study studies/models/best/mlp:lkrnnmlp --study studies/models/best/flow:lkrnnflow
python3 integral_res.py --input-pattern 'ttbar_output_best_{ch}.root' --methods lkr lkrnn lkrnnmlp lkrnnflow
python fold_check.py --input-pattern 'ttbar_output_best_{ch}.root'
```

Як це працює:
- **Входи.** Для LKRnn і моделей із вхідом `raw`/`features` усе потрібне вже є у файлі: 26 ознак `nn_features`
  (їх рахує `LKRv3::computeFeatures` у C++) і розв'язок LKRv3 (`mtt/pttt/ytt/phitt_lkrv3`). Моделям із CSV
  (`raw_csv_mlj` тощо) потрібен супутник `ttbar_output_extra_{ch}.root` поруч із вхідним файлом —
  `nn_apply` підхоплює його сам і перевіряє, що події збігаються (за `mtt_gen`).
- **`--det-model` / `--gen-model`** — гілки `lkrnn` / `lkrnn_gen` (робочий LKRnn, як у `LKRnn.cxx`):
  шлях до папки моделі або `keep` (не чіпати). За замовчуванням беруться `solve_nn_output` і
  `solve_nn_gen_output`; результат збігається з C++ до ~1e-6 (перевірка — `--check`).
- **`--study ПАПКА:НАЗВА`** — дослідницька модель `train_study.py` (MLP, Flow, LKRnn-варіанти); можна
  кілька разів. Для кожної з'являються гілки `НАЗВА` (1 — розв'язок є, 0 — ні) і `mtt_/pttt_/ytt_/phitt_/
  dphitt_НАЗВА`. Назва має починатися з `lkrnn`. Для Flow оцінку можна вибрати суфіксом папки:
  `ПАПКА@mean`, `@median`, `@mode` (за замовчуванням — та, з якою модель тренувалась; у `train_best.sh` — mean).
  Кут φ і Δφ мережі не передбачають — вони беруться з LKRv3.
- **`--study-cal ПАПКА:НАЗВА[:α]`** — те саме з калібруванням (`calibration.json` моделі), α — частка поправки.
- **Поділ train/val/test.** `nn_apply` перевіряє, що поділ моделі збігається з поділом LKRnn, і записує у
  вихідний файл службовий об'єкт `nn_apply_info` (які моделі, на якому рівні тренувались). З нього
  `integral_res.py`, `paper_figs.py`, `fold_check.py` та ін. знають, які події мережа бачила при тренуванні,
  і оцінюють її лише на решті (~15%), без додаткових аргументів.
- **Ланцюжок.** Вихід `nn_apply` можна подати йому ж на вхід, щоб додати ще моделей: попередні гілки
  копіюються, їхній опис у `nn_apply_info` переноситься.
- Потрібне середовище з torch (і zuko для Flow); на GPU MLP — секунди на канал, Flow — кілька хвилин.

---

- `--nngen 0` — детекторна модель (основні результати), `--nngen 1` — генераторна на детекторних даних;
  у конфізі це `krNNGen`, прапорець його перекриває.
- Алгоритми оцінюються на всьому датасеті, LKRnn — на подіях поза тренуванням (`nn_split.py`);
  ефективність — на всіх подіях.
- `nn_weights*.h` не комітити; після зміни моделі — `weights.py` + `./compile.sh`.
- Дослідницькі моделі й прогони — у `studies/` (`studies/README.md`), гілка з TMVA SOFIE — `lkrnn_sofie`.
