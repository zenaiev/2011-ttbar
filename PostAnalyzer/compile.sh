#!/bin/bash

# compile code (produces two executables)
KINRECOCODE='kinreco/FKR.cxx fkr/analysisUtils.cc fkr/KinematicReconstruction.cc fkr/KinematicReconstruction_LSroutines.cc fkr/KinematicReconstruction_MeanSol.cc fkr/KinematicReconstructionSolution.cc fkr/sampleHelpers.cc '
KINRECOCODE+='kinreco/SKR.cxx '
KINRECOCODE+='kinreco/LKR.cxx '
KINRECOCODE+='kinreco/LKRv2.cxx '
KINRECOCODE+='kinreco/LKRv3.cxx '
KINRECOCODE+='kinreco/LKRnn.cxx '
# -O2: оптимізація (нейромережа й реконструкції в рази швидші); TMVA SOFIE — читання ONNX-моделей LKRnn під час запуску
g++ -O2 $KINRECOCODE ttbarMakeHist.cxx read_config.cxx -o ttbarMakeHist `root-config --cflags --libs` -lMathMore -lTMVA -lROOTTMVASofie -lROOTTMVASofieParser -I.
g++ ttbarMakePlots.cxx -o ttbarMakePlots `root-config --cflags --libs`

# create needed directories if do not exist yet
mkdir -p data mc hist plots kr_performance
