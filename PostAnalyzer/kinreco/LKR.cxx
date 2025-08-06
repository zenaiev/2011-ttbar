#include "LKR.h"
#include "krvars.h"
#include "LooseKinReco.h"

LKR::LKR() : KinRecoBase("lkr") {}

std::vector<TLorentzVector> LKR::reconstruct(
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

bool LKR::selectBestJets(const TLorentzVector& vecLepM, const TLorentzVector& vecLepP,
      const std::vector<TLorentzVector>& vecJets, Float_t* jetBTagDiscr, const double bTagDiscrL,
      TLorentzVector& jetBest1, TLorentzVector& jetBest2) {
    // find best jets: prefer b-tagged jets, among those with equal b-tag number prefer jets with the highest sum of pT, require M(lb) < 180 GeV
  int bTagBest = 0;
  float pTSumBest = 0;
  bool status = false;
  for(std::vector<TLorentzVector>::const_iterator jet1 = vecJets.begin(); jet1 != vecJets.end(); jet1++)
  {
    if((*jet1+vecLepM).M() > 180 && (*jet1+vecLepP).M() > 180) continue;
    for(std::vector<TLorentzVector>::const_iterator jet2 = vecJets.begin(); jet2 != vecJets.end(); jet2++)
    {
      if((*jet2+vecLepM).M() > 180 && (*jet2+vecLepP).M() > 180) continue;
      // skip same jets
      if(jet1 == jet2) continue;
      // for this pair of jets, calculate number of b-tagged jets,
      // b-tagged jets are provided with negative masses (see selection.h)
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
      status = true;
    }
  }
  return status;
}

TLorentzVector LKR::solve(const TLorentzVector& lepton, const TLorentzVector& antilepton,
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
