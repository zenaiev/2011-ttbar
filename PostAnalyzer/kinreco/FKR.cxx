#include "FKR.h"
#include "fkr/KinematicReconstruction.h"
#include "fkr/KinematicReconstructionSolution.h"
#include "fkr/KinematicReconstruction_LSroutines.h"
#include "fkr/analysisUtils.h"

FKR::FKR() {
  int minBTag = 1;
  bool preferBtags = true;
  _kinReco = new KinematicReconstruction("fkr/KinRecoInput.root", minBTag, preferBtags);  
}

std::vector<TLorentzVector> FKR::reconstruct(
    const TLorentzVector& vecLepM, const TLorentzVector& vecLepP,
    const std::vector<TLorentzVector>& vecJets, Float_t* jetBTagDiscr, const double bTagDiscrL,
    const Float_t metPx, const Float_t metPy) {
  VLV leptons = {common::TLVtoLV(vecLepM), common::TLVtoLV(vecLepP)};
  std::vector<int> krLepInd = { 0 };
  std::vector<int> krAntiLepInd = { 1 };
  std::vector<int> krJetInd;
  std::vector<int> krBJetInd;
  VLV jets;
  std::vector<double> btags;
  for(int j = 0; j < vecJets.size(); j++) 
  {
    jets.push_back(common::TLVtoLV(vecJets[j]));
    btags.push_back(jetBTagDiscr[j]);
    krJetInd.push_back(j);
    if(jetBTagDiscr[j] > bTagDiscrL)
      krBJetInd.push_back(j);
  }
  TLorentzVector met;
  met.SetPx(metPx);
  met.SetPy(metPy);
  KinematicReconstructionSolutions krSolutions = _kinReco->solutions(krLepInd, krAntiLepInd, krJetInd, krBJetInd, leptons, jets, btags, common::TLVtoLV(met));
  bool flagPassedKinRec = krSolutions.numberOfSolutions();
  std::vector<TLorentzVector> solution;
  if (flagPassedKinRec) {
    solution.push_back(common::LVtoTLV(krSolutions.solution().top())); // t
    solution.push_back(common::LVtoTLV(krSolutions.solution().antiTop())); // tbar
    solution.push_back(solution[0] + solution[1]); // ttbar
  }
  return solution;
}
