TLorentzVector LooseKinReco(const TLorentzVector& lepton, const TLorentzVector& antilepton,
                                           const TLorentzVector& bjet, const TLorentzVector& bbarjet,
                                           float met_x, float met_y) {

TLorentzVector nunubar;
nunubar.SetPx(met_x);
nunubar.SetPy(met_y);
TLorentzVector ttbar=lepton+antilepton+bjet+bbarjet+nunubar;
return ttbar;
}