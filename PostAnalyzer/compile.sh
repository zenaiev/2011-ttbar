#!/bin/bash

# compile code (produces two executables)
#g++ ttbarMakeHist.cxx `root-config --cflags --libs` -lMathMore -o ttbarMakeHist
g++ -o ttbarMakeHist `root-config --cflags --libs` -lMathMore ttbarMakeHist.cxx fkr/analysisUtils.cc fkr/KinematicReconstruction.cc fkr/KinematicReconstruction_LSroutines.cc fkr/KinematicReconstruction_MeanSol.cc fkr/KinematicReconstructionSolution.cc fkr/sampleHelpers.cc
g++ ttbarMakePlots.cxx `root-config --cflags --libs` -lMathMore -o ttbarMakePlots

# create needed directories if do not exist yet
mkdir -p data mc hist plots
