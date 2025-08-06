#!/bin/bash
start=$(date +%s.%N)
rm -rf kr_performance/*
./ttbarMakeHist config_test.txt
python kinreco_eff_v3.py ttbar_output_3.root
python compare_output.py kr_performance ref_ev10000_ch3
end=$(date +%s.%N)
runtime=$(echo "$end - $start" | bc)
runtime=$(printf "%.0f" "$runtime")
echo "It took $runtime seconds"
