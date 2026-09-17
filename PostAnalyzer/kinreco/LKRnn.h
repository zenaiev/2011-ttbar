#pragma once

#include "LKRv3.h"
#include <memory>
#include <string>
#include <vector>
#include <TLorentzVector.h>

namespace TMVA { namespace Experimental { class RSofieReader; } }

// LKRnn: нейромережева поправка до розв'язку LKRv3::solve() для M, pT і y системи tt̄.
// Мережа (ONNX разом із нормуванням входу) читається під час запуску з <modelDir>/solve_nn.onnx
// через ROOT TMVA SOFIE, тож нова модель не потребує перекомпіляції. Ознаки (LKRv3::computeFeatures)
// і перетворення поправок у tt̄ лишаються в C++.
class LKRnn : public LKRv3 {
  public:
    explicit LKRnn(const std::string& modelDir);
    ~LKRnn();

    virtual std::vector<TLorentzVector> reconstruct(
      const TLorentzVector& vecLepM, const TLorentzVector& vecLepP,
      const std::vector<TLorentzVector>& vecJets,
      Float_t* jetBTagDiscr, const double bTagDiscrL,
      const Float_t metPx, const Float_t metPy
    ) override;

  private:
    std::shared_ptr<TMVA::Experimental::RSofieReader> _reader;   // спільний на процес (кеш у LKRnn.cxx)
};
