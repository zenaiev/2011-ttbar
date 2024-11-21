#!/bin/bash

# compile code (produces two executables)
g++ ttbarMakeHist.cxx `root-config --cflags --libs` -lMathMore -o ttbarMakeHist
g++ ttbarMakePlots.cxx `root-config --cflags --libs` -lMathMore -o ttbarMakePlots

# create needed directories if do not exist yet
mkdir -p data mc hist plots
