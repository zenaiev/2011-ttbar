#!/bin/bash

# compile code (produces two executables)
g++ -o ttbarMakeHist `root-config --cflags --libs` -lMathMore ttbarMakeHist.cxx
g++ -o ttbarMakePlots `root-config --cflags --libs` ttbarMakePlots.cxx

# create needed directories if do not exist yet
mkdir -p data mc hist plots
