#pragma once

#include "LKRv3.h"
#include <vector>
#include <TLorentzVector.h>

// LKRnn v2: нейромережева заміна LKRv3::solve()
// Вихід мережі: {mtt, pttt, ytt, sin(phitt), cos(phitt)}
// ttbar TLorentzVector будується з цих 5 змінних.

class LKRnn : public LKRv3 {
  public:
    LKRnn();

    virtual std::vector<TLorentzVector> reconstruct(
      const TLorentzVector& vecLepM, const TLorentzVector& vecLepP,
      const std::vector<TLorentzVector>& vecJets,
      Float_t* jetBTagDiscr, const double bTagDiscrL,
      const Float_t metPx, const Float_t metPy
    ) override;

  private:
    // Нормалізація входу (18) і виходу (5) — заповнюються в конструкторі з nn_weights.h
    std::vector<float> x_mean, x_std;
    std::vector<float> y_mean, y_std;
};