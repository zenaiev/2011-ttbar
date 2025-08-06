#!/bin/bash
start=$(date +%s.%N)
./ttbarMakeHist
python kinreco_eff_v3.py ttbar_output_3.root
python compare_output.py
end=$(date +%s.%N)
runtime=$(echo "$end - $start" | bc)
runtime=$(printf "%.0f" "$runtime")
echo "It took $runtime seconds"
