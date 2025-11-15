#include "LKRv2.h"
#include "krvars.h"

LKRv2::LKRv2() : KinRecoBase("lkrv2") {}

std::vector<TLorentzVector> LKRv2::reconstruct(
    const TLorentzVector& vecLepM, const TLorentzVector& vecLepP,
    const std::vector<TLorentzVector>& vecJets, Float_t* jetBTagDiscr, const double bTagDiscrL,
    const Float_t metPx, const Float_t metPy) {
  std::vector<TLorentzVector> solution;
  TLorentzVector jetBest1, jetBest2;
  bool found = selectBestJets(vecLepM, vecLepP, vecJets, jetBTagDiscr, bTagDiscrL, jetBest1, jetBest2);
  if (found) {
    if (solution.size() == 0) {
      solution.resize(3);
    }
    solution[2] = solve(vecLepM, vecLepP, jetBest1, jetBest2, metPx, metPy);
  }
  return solution;
}
bool LKRv2::selectBestJets(
    const TLorentzVector& vecLepM, const TLorentzVector& vecLepP,
    const std::vector<TLorentzVector>& vecJets, Float_t* jetBTagDiscr,
    const double bTagDiscrL, TLorentzVector& jetBest1, TLorentzVector& jetBest2) {

    int bTagBest = 0;
    float pTSumBest = 0.;
    bool foundPair = false;

    for (auto jet1 = vecJets.begin(); jet1 != vecJets.end(); ++jet1) {

        bool flagLepM1 = ((*jet1 + vecLepM).M() < 180);
        bool flagLepP1 = ((*jet1 + vecLepP).M() < 180);

        if (!(flagLepM1 || flagLepP1)) continue;  

        for (auto jet2 = vecJets.begin(); jet2 != vecJets.end(); ++jet2) {
            if (jet1 == jet2) continue;

            bool flagLepM2 = ((*jet2 + vecLepM).M() < 180);
            bool flagLepP2 = ((*jet2 + vecLepP).M() < 180);

            if (!(flagLepM2 || flagLepP2)) continue; 
       
              // обрахунок кількості true для усіх флагів
            int nTrue = int(flagLepM1) + int(flagLepP1) + int(flagLepM2) + int(flagLepP2);
            bool rejectedByLKRv2 = false;
            // Пошук true значень при nTrue = 2
            if (nTrue == 2) {
                bool sameJet1 = (flagLepM1 && flagLepP1); // обидва true для jet1
                bool sameJet2 = (flagLepM2 && flagLepP2); // обидва true для jet2
                bool sameLepM = (flagLepM1 && flagLepM2); // обидва для lepM
                bool sameLepP = (flagLepP1 && flagLepP2); // обидва для lepP

                // перевірка й відкидання випадку, де true належать одному джету або одному лептону./
                if (sameJet1 || sameJet2 || sameLepM || sameLepP)
                    continue;
                    //rejectedByLKRv2 = true;
            }
            //if (!rejectedByLKRv2) continue;

            int bTag = int(jetBTagDiscr[jet1 - vecJets.begin()] > bTagDiscrL)
                     + int(jetBTagDiscr[jet2 - vecJets.begin()] > bTagDiscrL);

            if (bTag < bTagBest) continue;
            if (bTag > bTagBest) pTSumBest = 0.;
            bTagBest = bTag;

            float pTSum = jet1->Pt() + jet2->Pt();
            if (pTSum < pTSumBest) continue;
            pTSumBest = pTSum;
            
            jetBest1 = *jet1;
            jetBest2 = *jet2;
            foundPair = true;
        }
    }

    return foundPair;
}

TLorentzVector LKRv2::solve(const TLorentzVector& lepton, const TLorentzVector& antilepton,
                                           const TLorentzVector& bjet, const TLorentzVector& bbarjet,
                                           float met_x, float met_y) {
  TLorentzVector llbar = lepton + antilepton;
  TLorentzVector nunubar;
  nunubar.SetPx(met_x);
  nunubar.SetPy(met_y);
  if(nunubar.Pt() < llbar.E()) {
    nunubar.SetPz(llbar.Pz());
  }
  else {
    nunubar.SetPz(0.);
  }
  if(nunubar.P() < llbar.E()) {
    nunubar.SetE(llbar.E());
  }
  else {
    nunubar.SetE(nunubar.P());
  }
  assert(nunubar.M()>=-0.001);
  TLorentzVector llnn = llbar + nunubar;
  const double mw = 80.4;
  if(llnn.M() < (2.0 * mw)) {
    const double eNew = TMath::Sqrt(4 * mw * mw + llnn.Perp2()) * TMath::CosH(llnn.Rapidity());
    const double zNew =eNew * TMath::TanH(llnn.Rapidity());
    llnn = TLorentzVector(llnn.X(), llnn.Y(), zNew, eNew);
  }
  TLorentzVector ttbar = llnn + bjet + bbarjet;
  return ttbar;
}