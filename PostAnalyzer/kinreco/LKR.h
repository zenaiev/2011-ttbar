
#pragma once

#include "KinRecoBase.h"

class LKR: public KinRecoBase {
  public:
    LKR();
    virtual std::vector<TLorentzVector> reconstruct(
      const TLorentzVector& vecLepM, const TLorentzVector& vecLepP,
      const std::vector<TLorentzVector>& vecJets, Float_t* jetBTagDiscr, const double bTagDiscrL,
      const Float_t metPx, const Float_t metPy
    );
  protected:
    virtual bool selectBestJets(const TLorentzVector& vecLepM, const TLorentzVector& vecLepP,
      const std::vector<TLorentzVector>& vecJets, Float_t* jetBTagDiscr, const double bTagDiscrL,
      TLorentzVector& jetBest1, TLorentzVector& jetBest2);
    virtual TLorentzVector solve(const TLorentzVector& lepton, const TLorentzVector& antilepton,
      const TLorentzVector& bjet, const TLorentzVector& bbarjet, float met_x, float met_y);
};
