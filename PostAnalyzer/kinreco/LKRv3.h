#pragma once

#include "KinRecoBase.h"

class LKRv3: public KinRecoBase {
  public:
    LKRv3();
    LKRv3(const std::string& name); // named instance (e.g. for subclasses like LKRnn)
    virtual std::vector<TLorentzVector> reconstruct(
      const TLorentzVector& vecLepM, const TLorentzVector& vecLepP,
      const std::vector<TLorentzVector>& vecJets, Float_t* jetBTagDiscr, const double bTagDiscrL,
      const Float_t metPx, const Float_t metPy
    );
    // публічна обгортка над selectBestJets (для виклику з eventReco.h), як у LKR
    bool selectBestJetsPublic(const TLorentzVector& vecLepM, const TLorentzVector& vecLepP,
      const std::vector<TLorentzVector>& vecJets, Float_t* jetBTagDiscr, const double bTagDiscrL,
      TLorentzVector& jetBest1, TLorentzVector& jetBest2) {
        return selectBestJets(vecLepM, vecLepP, vecJets, jetBTagDiscr, bTagDiscrL, jetBest1, jetBest2);
      }
    // 26 сирих (ненормалізованих) ознак для нейромережі LKRnn — з ВЖЕ відібраних джетів j1, j2.
    // Єдина реалізація ознак: використовується і LKRnn (inference), і eventReco (дамп у дерево).
    std::vector<float> computeFeatures(
      const TLorentzVector& vecLepM, const TLorentzVector& vecLepP,
      const TLorentzVector& j1, const TLorentzVector& j2,
      const Float_t metPx, const Float_t metPy);
  protected:
    virtual bool selectBestJets(const TLorentzVector& vecLepM, const TLorentzVector& vecLepP,
      const std::vector<TLorentzVector>& vecJets, Float_t* jetBTagDiscr, const double bTagDiscrL,
      TLorentzVector& jetBest1, TLorentzVector& jetBest2);
    virtual TLorentzVector solve(const TLorentzVector& lepton, const TLorentzVector& antilepton,
      const TLorentzVector& bjet, const TLorentzVector& bbarjet, float met_x, float met_y);
};