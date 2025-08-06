
#pragma once

#include "KinRecoBase.h"
class KinematicReconstruction;

class FKR: public KinRecoBase {
  public:
    FKR();
    virtual std::vector<TLorentzVector> reconstruct(
      const TLorentzVector& vecLepM, const TLorentzVector& vecLepP,
      const std::vector<TLorentzVector>& vecJets, Float_t* jetBTagDiscr, const double bTagDiscrL,
      const Float_t metPx, const Float_t metPy
    );
  
  protected:
    KinematicReconstruction* _kinReco = nullptr;
};
