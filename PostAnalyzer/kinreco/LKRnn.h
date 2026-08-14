#pragma once

#include "LKRv3.h"
#include <vector>
#include <TLorentzVector.h>

// LKRnn v2: нейромережева заміна LKRv3::solve()
// Вихід мережі: {mtt, pttt, ytt, sin(phitt), cos(phitt)}
// ttbar TLorentzVector будується з цих 5 змінних.

// набір вказівників на ваги однієї моделі, заповнюється в конструкторі з
// відповідного namespace (NNWeights = детекторна, NNWeights_gen = генераторна)
struct NNParams {
  const std::vector<float> *ip0w, *ip0b, *ip1w, *ip1b;
  const std::vector<float> *b0_0w, *b0_0b, *b0_1w, *b0_1b, *b0_3w, *b0_3b, *b0_4w, *b0_4b;
  const std::vector<float> *b1_0w, *b1_0b, *b1_1w, *b1_1b, *b1_3w, *b1_3b, *b1_4w, *b1_4b;
  const std::vector<float> *b2_0w, *b2_0b, *b2_1w, *b2_1b, *b2_3w, *b2_3b, *b2_4w, *b2_4b;
  const std::vector<float> *hw, *hb;
};

class LKRnn : public LKRv3 {
  public:
    // useGen=false -> детекторна модель (NNWeights); true -> генераторна (NNWeights_gen)
    LKRnn(bool useGen = false);

    virtual std::vector<TLorentzVector> reconstruct(
      const TLorentzVector& vecLepM, const TLorentzVector& vecLepP,
      const std::vector<TLorentzVector>& vecJets,
      Float_t* jetBTagDiscr, const double bTagDiscrL,
      const Float_t metPx, const Float_t metPy
    ) override;

  private:
    std::vector<float> x_mean, x_std;  // нормалізація входу з обраної моделі
    NNParams _p;                        // вказівники на ваги обраної моделі
};