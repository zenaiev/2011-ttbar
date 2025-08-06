#!/bin/bash

# compile code (produces two executables)
KINRECOCODE='kinreco/FKR.cxx fkr/analysisUtils.cc fkr/KinematicReconstruction.cc fkr/KinematicReconstruction_LSroutines.cc fkr/KinematicReconstruction_MeanSol.cc fkr/KinematicReconstructionSolution.cc fkr/sampleHelpers.cc '
KINRECOCODE+='kinreco/SKR.cxx '
KINRECOCODE+='kinreco/LKR.cxx '
g++ $KINRECOCODE ttbarMakeHist.cxx read_config.cxx -o ttbarMakeHist `root-config --cflags --libs` -lMathMore -I.
#g++ ttbarMakePlots.cxx -o ttbarMakePlots `root-config --cflags --libs`

# create needed directories if do not exist yet
mkdir -p data mc hist plots kr_performance
