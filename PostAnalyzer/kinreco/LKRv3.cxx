#include "LKRv3.h"
#include "krvars.h"
#include <cmath>

LKRv3::LKRv3() : KinRecoBase("lkrv3") {}
LKRv3::LKRv3(const std::string& name) : KinRecoBase(name) {}

// 26 сирих ознак для LKRnn з відібраних джетів j1, j2 (винесено з LKRnn.cxx,
// щоб inference і дамп у eventReco використовували ОДНУ реалізацію).
std::vector<float> LKRv3::computeFeatures(
    const TLorentzVector& vecLepM, const TLorentzVector& vecLepP,
    const TLorentzVector& j1, const TLorentzVector& j2,
    const Float_t metPx, const Float_t metPy) {
    // ── 1. Скалярні змінні першої групи ───────────────────────────
    float m_lpj1 = (vecLepP + j1).M();
    float m_lmj2 = (vecLepM + j2).M();
    float ht = vecLepM.Pt() + vecLepP.Pt() + j1.Pt() + j2.Pt()
               + std::sqrt(metPx*metPx + metPy*metPy);

    // ── 2. LKRv3-подібні фізичні constraints ──────────────────────
    TLorentzVector llbar = vecLepM + vecLepP;
    float llbar_m   = llbar.M();
    float llbar_e   = llbar.E();
    float llbar_pz  = llbar.Pz();

    float ep = llbar_e + llbar_pz;
    float em = llbar_e - llbar_pz;
    float llbar_rap = (ep > 0.f && em > 0.f) ? 0.5f * std::log(ep / em) : 0.f;

    float met_pt  = std::sqrt(metPx*metPx + metPy*metPy);
    float mt_nunu = std::sqrt(llbar_m*llbar_m + met_pt*met_pt);
    float pz_nunu_lkr = mt_nunu * std::sinh(llbar_rap);
    float e_nunu_lkr  = mt_nunu * std::cosh(llbar_rap);

    float llnn_px = llbar.Px() + metPx;
    float llnn_py = llbar.Py() + metPy;
    float llnn_pz = llbar_pz   + pz_nunu_lkr;
    float llnn_e  = llbar_e    + e_nunu_lkr;
    float llnn_m2 = llnn_e*llnn_e - llnn_px*llnn_px - llnn_py*llnn_py - llnn_pz*llnn_pz;
    float llnn_m  = llnn_m2 > 0.f ? std::sqrt(llnn_m2) : 0.f;

    // ── 3. Вхідний вектор (РІВНО 26 ОЗНАК) ─────────────────────────
    return {
        (float)vecLepM.E(),  (float)vecLepM.Px(), (float)vecLepM.Py(), (float)vecLepM.Pz(),
        (float)vecLepP.E(),  (float)vecLepP.Px(), (float)vecLepP.Py(), (float)vecLepP.Pz(),
        (float)j1.E(),  (float)j1.Px(), (float)j1.Py(), (float)j1.Pz(),
        (float)j2.E(),  (float)j2.Px(), (float)j2.Py(), (float)j2.Pz(),
        metPx, metPy,
        m_lpj1, m_lmj2, ht,
        llbar_m, llbar_rap, mt_nunu, pz_nunu_lkr, llnn_m
    };
}

std::vector<TLorentzVector> LKRv3::reconstruct(
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
bool LKRv3::selectBestJets(
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
            bool rejectedByLKRv2 = true;
            if (!(flagLepM2 || flagLepP2)) continue; 
            
              // обрахунок кількості true для усіх флагів
            int nTrue = int(flagLepM1) + int(flagLepP1) + int(flagLepM2) + int(flagLepP2);
            // Пошук true значень при nTrue = 2
            if (nTrue == 2) {
                bool sameJet1 = (flagLepM1 && flagLepP1); // обидва true для jet1
                bool sameJet2 = (flagLepM2 && flagLepP2); // обидва true для jet2
                bool sameLepM = (flagLepM1 && flagLepM2); // обидва для lepM
                bool sameLepP = (flagLepP1 && flagLepP2); // обидва для lepP

                // перевірка й відкидання випадку, де true належать одному джету або одному лептону./
                  if (sameJet1 || sameJet2 || sameLepM || sameLepP)
                        continue;
                
            }

        int bTag = int(jet1->M() < 0) + int(jet2->M() < 0);
        if(bTag < bTagBest) continue;
        // need to reset the best pT sum to 0 if we found a higher b-tag category
        if(bTag > bTagBest) pTSumBest = 0.;
        bTagBest = bTag;
        // calculate the sum of pT
        float pTSum = jet1->Pt() + jet2->Pt();
        if(pTSum < pTSumBest) continue;
        pTSumBest = pTSum;
        // store these jets
        if(jet1->M() < 0) jetBest1.SetPtEtaPhiM(jet1->Pt(), jet1->Eta(), jet1->Phi(), -1 * jet1->M());
        jetBest1 = *jet1;
        if(jet2->M() < 0) jetBest2.SetPtEtaPhiM(jet2->Pt(), jet2->Eta(), jet2->Phi(), -1 * jet2->M());
        else jetBest2 = *jet2;
        foundPair = true;
        }
    }

    return foundPair;
}

TLorentzVector LKRv3::solve(const TLorentzVector& lepton, const TLorentzVector& antilepton,
                          const TLorentzVector& bjet, const TLorentzVector& bbarjet,
                          float met_x, float met_y) {
  TLorentzVector llbar = lepton + antilepton;
  TLorentzVector nunubar;
  
  nunubar.SetPx(met_x);
  nunubar.SetPy(met_y);

  double mass_ll = llbar.M();
  double rap_ll = llbar.Rapidity();
  
  double pt_nunu = nunubar.Pt();
  double mt_nunu = TMath::Sqrt(mass_ll * mass_ll + pt_nunu * pt_nunu);

  double pz_nunu = mt_nunu * TMath::SinH(rap_ll);
  double e_nunu  = mt_nunu * TMath::CosH(rap_ll);

  nunubar.SetPz(pz_nunu);
  nunubar.SetE(e_nunu);

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