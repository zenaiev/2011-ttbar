#!/bin/bash
# Паралельний запуск ttbarMakeHist по каналах розпаду.
# Кожен канал (ee/mumu/emu) — окремий процес зі своїм однопанальним конфігом,
# пише в окремий ttbar_output_full_{1,2,3}.root -> конфліктів немає.
# Прискорення ~ до 3× на 3+ ядрах (обмежене найбільшим каналом, зазвичай emu).
#
# Використання:
#   ./run_full_parallel.sh [config_full_output.txt]
# --- аргументи -------------------------------------------------------------
CONFIG="config_full_output.txt"
NNGEN=""
while [ $# -gt 0 ]; do
  case "$1" in
    --nngen) NNGEN="$2"; shift 2 ;;
    -h|--help)
      echo "Використання: $0 [конфіг] [--nngen 0|1]"
      echo "  --nngen 0  детекторна NN-модель (основні результати)"
      echo "  --nngen 1  генераторна NN-модель на детекторних даних (крос-тест)"
      exit 0 ;;
    -*) echo "[E] невідома опція: $1"; exit 1 ;;
    *)  CONFIG="$1"; shift ;;
  esac
done

if [ ! -f "$CONFIG" ]; then
  echo "[E] конфіг не знайдено: $CONFIG"; exit 1
fi
if [ -n "$NNGEN" ] && [ "$NNGEN" != "0" ] && [ "$NNGEN" != "1" ]; then
  echo "[E] --nngen приймає лише 0 або 1 (отримано: $NNGEN)"; exit 1
fi
[ -n "$NNGEN" ] && echo "[I] krNNGen=$NNGEN ($([ "$NNGEN" = 1 ] && echo 'ГЕНЕРАТОРНА модель на детекторних даних' || echo 'детекторна модель'))"

# які канали увімкнені у конфізі
declare -A CH=( [1]=channel_ee [2]=channel_mumu [3]=channel_emu )
enabled=()
for ch in 1 2 3; do
  val=$(awk -v k="${CH[$ch]}" '$1==k{print $2}' "$CONFIG")
  [ "$val" = "1" ] && enabled+=("$ch")
done
echo "[I] Канали до обробки: ${enabled[*]}"

pids=()
for ch in "${enabled[@]}"; do
  cfg="config_run_c${ch}.txt"
  # копія конфіга з увімкненим лише одним каналом
  awk -v c="$ch" -v g="$NNGEN" '
    $1=="channel_ee"   {print "channel_ee "   (c==1?1:0); next}
    $1=="channel_mumu" {print "channel_mumu " (c==2?1:0); next}
    $1=="channel_emu"  {print "channel_emu "  (c==3?1:0); next}
    $1=="krNNGen" && g!="" {print "krNNGen " g; seen=1; next}
    {print}
    END {if (g!="" && !seen) print "krNNGen " g}
  ' "$CONFIG" > "$cfg"
  echo "[I] Запуск каналу $ch -> лог full_run_c${ch}.log"
  ./ttbarMakeHist "$cfg" > "full_run_c${ch}.log" 2>&1 &
  pids+=($!)
done

# чекаємо всі процеси, фіксуємо, чи всі успішні
fail=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "[OK]   канал ${enabled[$i]} завершено"
  else
    echo "[FAIL] канал ${enabled[$i]} впав (див. full_run_c${enabled[$i]}.log)"
    fail=1
  fi
done

# прибираємо тимчасові конфіги
rm -f config_run_c*.txt
[ $fail -eq 0 ] && echo "[I] Усі канали завершено успішно." || echo "[E] Були помилки — перевірте логи."
exit $fail
