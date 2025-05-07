TLorentzVector ConstrainM(const TLorentzVector& lv, const double m)
{
  if(lv.M() > m)
    throw(std::logic_error("Error in ZNtupleAnalysis::ConstrainM(): lv.M() = %f > %f = m"), lv.M(), m);
  const double eNew = TMath::Sqrt(m * m + lv.Perp2()) * TMath::CosH(lv.Rapidity());
  const double zNew =eNew * TMath::TanH(lv.Rapidity());
  TLorentzVector lvNew(lv.X(), lv.Y(), zNew, eNew);
  assert(TMath::Abs(lvNew.M() - m) < 0.1);
  return lvNew;
}


TLorentzVector LooseKinReco(const TLorentzVector& lepton, const TLorentzVector& antilepton,
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
    const double eNew = TMath::Sqrt(mw * mw + llnn.Perp2()) * TMath::CosH(llnn.Rapidity());
    const double zNew =eNew * TMath::TanH(llnn.Rapidity());
    llnn = TLorentzVector(llnn.X(), llnn.Y(), zNew, eNew);
  }

  TLorentzVector ttbar = llnn + bjet + bbarjet;
  return ttbar;
}