#!/bin/bash
# Паралельний запуск ttbarMakeHist по каналах розпаду.
# Кожен канал (ee/mumu/emu) — окремий процес зі своїм однопанальним конфігом,
# пише в окремий ttbar_output_full_{1,2,3}.root -> конфліктів немає.
# Прискорення ~ до 3× на 3+ ядрах (обмежене найбільшим каналом, зазвичай emu).
#
# Використання:
#   ./run_full_parallel.sh [config_full_output.txt]
CONFIG="${1:-config_full_output.txt}"

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
  awk -v c="$ch" '
    $1=="channel_ee"   {print "channel_ee "   (c==1?1:0); next}
    $1=="channel_mumu" {print "channel_mumu " (c==2?1:0); next}
    $1=="channel_emu"  {print "channel_emu "  (c==3?1:0); next}
    {print}
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
