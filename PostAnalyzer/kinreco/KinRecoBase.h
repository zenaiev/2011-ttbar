#pragma once

#include "kinreco/krvars.h"
#include <vector>
#include <TLorentzVector.h>
#include <TTree.h>

class KinRecoBase {
  public:
    std::string GetName() {return _name;}

    void init(TTree* tree, const std::vector<KRVAR*>& krvars) {
      _krvars = krvars;
      _tree_vars.resize(_krvars.size());
      for(size_t i = 0; i < _krvars.size(); i++) {
        auto name = _krvars[i]->GetName() + "_" + _name;
        tree->Branch(name.c_str(), &_tree_vars[i], (name + "/F").c_str());
      }
    }

    void reset_vars() {
      for(size_t i = 0; i < _krvars.size(); i++) {
        _tree_vars[i] = -1000.0;
      }
    }

    void calculate_vars(const TLorentzVector& t, const TLorentzVector& tbar, const TLorentzVector& ttbar) {
      for(size_t i = 0; i < _krvars.size(); i++) {
        _tree_vars[i] = _krvars[i]->calculate(t, tbar, ttbar);
      }
    }

    virtual std::vector<TLorentzVector> reconstruct(
      const TLorentzVector& vecLepM, const TLorentzVector& vecLepP,
      const std::vector<TLorentzVector>& vecJets, Float_t* jetBTagDiscr, const double bTagDiscrL,
      const Float_t metPx, const Float_t metPy
    ) = 0;

  protected:
    KinRecoBase(const std::string& name) : _name(name) {} // Protected constructor to initialize _name
    const std::string _name;
    std::vector<KRVAR*> _krvars;
    std::vector<float> _tree_vars;
};
