
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
    const double eNew = TMath::Sqrt(4 * mw * mw + llnn.Perp2()) * TMath::CosH(llnn.Rapidity());
    const double zNew =eNew * TMath::TanH(llnn.Rapidity());
    llnn = TLorentzVector(llnn.X(), llnn.Y(), zNew, eNew);
  }
  printf("l1 ln: %+7.0f%+7.0f%+7.0f%+7.0f\n", llnn.Px(), llnn.Py(), llnn.Pz(), llnn.E());
  printf("l1 b : %+7.0f%+7.0f%+7.0f%+7.0f\n", (bjet).Px(), (bjet).Py(), (bjet).Pz(), (bjet).E());
  printf("l1 bb: %+7.0f%+7.0f%+7.0f%+7.0f\n", (bbarjet).Px(), (bbarjet).Py(), (bbarjet).Pz(), (bbarjet).E());
  TLorentzVector ttbar = llnn + bjet + bbarjet;
  return ttbar;
}

TLorentzVector LooseKinReco_mod(const TLorentzVector& lepton, const TLorentzVector& antilepton,
  const TLorentzVector& bjet, const TLorentzVector& bbarjet,
  float met_x, float met_y) {
  const TLorentzVector v0;
  TLorentzVector llbar = lepton + antilepton;
  TLorentzVector nunubar;
  nunubar.SetPx(met_x);
  nunubar.SetPy(met_y);
  //double scale = TMath::Sqrt(met_x * met_x + met_y * met_y) / llbar.Pt();
  double scale = 1.0;
  if(nunubar.Pt() < llbar.E()) {
    nunubar.SetPz(llbar.Pz() * scale);
  }
  else {
    return v0;
    nunubar.SetPz(0.);
  }
  if(nunubar.P() < llbar.E()) {
    nunubar.SetE(llbar.E() * scale);
  }
  else {
    return v0;
    nunubar.SetE(nunubar.P());
  }
  assert(nunubar.M()>=-0.001);
  TLorentzVector llnn = llbar + nunubar;
  const double mw = 80.4;
  if(llnn.M() < (2.0 * mw)) {
  //if(llnn.M() < 0.7 * (2.0 * mw)) {
    return v0;
    const double eNew = TMath::Sqrt(4 * mw * mw + llnn.Perp2()) * TMath::CosH(llnn.Rapidity());
    const double zNew =eNew * TMath::TanH(llnn.Rapidity());
    llnn = TLorentzVector(llnn.X(), llnn.Y(), zNew, eNew);
  }
  printf("l2 ln: %+7.0f%+7.0f%+7.0f%+7.0f\n", llnn.Px(), llnn.Py(), llnn.Pz(), llnn.E());
  printf("l2 b : %+7.0f%+7.0f%+7.0f%+7.0f\n", (bjet).Px(), (bjet).Py(), (bjet).Pz(), (bjet).E());
  printf("l2 bb: %+7.0f%+7.0f%+7.0f%+7.0f\n", (bbarjet).Px(), (bbarjet).Py(), (bbarjet).Pz(), (bbarjet).E());
  TLorentzVector ttbar = llnn + bjet + bbarjet;
  return ttbar;
}
