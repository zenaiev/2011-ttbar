#pragma once
#include <string>
//class TLorentzVector;
#include <TLorentzVector.h>

class KRVAR {
  public:
    std::string GetName() {return _name;}
    virtual float calculate(const TLorentzVector& t, const TLorentzVector& tbar, const TLorentzVector& ttbar) = 0;
  protected:
    const std::string _name;
    KRVAR(const std::string& name) : _name(name) {} // Protected constructor to initialize _name
};

class Mtt : public KRVAR {
  public:
    Mtt() : KRVAR("mtt") {}
    virtual float calculate(const TLorentzVector& t, const TLorentzVector& tbar, const TLorentzVector& ttbar) {
      return ttbar.M();
    }
};

class Ytt : public KRVAR {
  public:
    Ytt() : KRVAR("ytt") {}
    virtual float calculate(const TLorentzVector& t, const TLorentzVector& tbar, const TLorentzVector& ttbar) {
      return ttbar.Rapidity();
    }
};

class Pttt : public KRVAR {
  public:
    Pttt() : KRVAR("pttt") {}
    virtual float calculate(const TLorentzVector& t, const TLorentzVector& tbar, const TLorentzVector& ttbar) {
      return ttbar.Pt();
    }
};

class Phitt : public KRVAR {
  public:
    Phitt() : KRVAR("phitt") {}
    virtual float calculate(const TLorentzVector& t, const TLorentzVector& tbar, const TLorentzVector& ttbar) {
      return ttbar.Phi();
    }
};
