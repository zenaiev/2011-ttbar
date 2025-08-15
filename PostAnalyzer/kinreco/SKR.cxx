#include "SKR.h"
#include "krvars.h"
#include "kinReco.h"

SKR::SKR() : KinRecoBase("skr") {}

std::vector<TLorentzVector> SKR::reconstruct(
    const TLorentzVector& vecLepM, const TLorentzVector& vecLepP,
    const std::vector<TLorentzVector>& vecJets, Float_t* jetBTagDiscr, const double bTagDiscrL,
    const Float_t metPx, const Float_t metPy) {
  TLorentzVector t, tbar;
  int status = KinRecoDilepton(vecLepM, vecLepP, vecJets, metPx, metPy, t, tbar);
  std::vector<TLorentzVector> solution;
  if (status) {
    solution.push_back(t); // t
    solution.push_back(tbar); // tbar
    solution.push_back(solution[0] + solution[1]); // ttbar
  }
  return solution;
}
