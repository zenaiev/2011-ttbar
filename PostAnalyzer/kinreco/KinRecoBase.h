#pragma once

#include <vector>
#include <TLorentzVector.h>

class KinRecoBase {
  public:
    virtual std::vector<TLorentzVector> reconstruct(
      const TLorentzVector& vecLepM, const TLorentzVector& vecLepP,
      const std::vector<TLorentzVector>& vecJets, Float_t* jetBTagDiscr, const double bTagDiscrL,
      const Float_t metPx, const Float_t metPy
    ) = 0;
};
