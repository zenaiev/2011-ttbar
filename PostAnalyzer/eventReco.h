// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
// >>>>>>>>>>>>>> Helper for ttbar event reconstruction >>>>>>>>>>>>>>>>
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

#ifndef TTBAR_EVENTRECO_H
#define TTBAR_EVENTRECO_H

// additional files from this analysis 
#include "tree.h"
#include "selection.h"
#include "settings.h"
#include "read_config.h"
// C++ library or ROOT header files
#include <map>
#include <TChain.h>
#include <TCanvas.h>
#include <TFile.h>
#include <TH1.h>

// kinematic reconstruction methods
#include "kinreco/FKR.h"
#include "kinreco/SKR.h"
#include "kinreco/LKR.h"
#include "kinreco/LKRv2.h"
#include "kinreco/LKRv3.h"
#include "kinreco/LKRnn.h"
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
// >>>>>>>>>>>>>>>>>>>>>>>> ZVarHisto class >>>>>>>>>>>>>>>>>>>>>>>>>>>>
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
//
// Class for control plot and cross section histograms: 
// it stores variable name and histogram. 
// A bunch of histograms can be filled using proper ttbar kinematics 
// input with just one line (see void FillHistos() below).
// Also see void StoreHistos() for histogram storage.
//
class ZVarHisto
{
  private:
    TH1D* zHisto; // histogram
    TString zVar; // variable name (see void FillHistos() fot its usage)
  
  public:
    // constructor
    ZVarHisto(const TString& str, TH1D* h)
    {
      zHisto = h;
      zVar = str;
    }

    // copy constructor
    ZVarHisto(const ZVarHisto& old)
    {
      zHisto = new TH1D(*(old.zHisto));
      zVar = old.zVar;
    }

    // access histogram
    TH1* H() {return zHisto;}
    
    // access variable name
    TString V() {return zVar;}
};
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
// >>>>>>>>>> Fill histogram from ZVarHisto class >>>>>>>>>>>>>>>>>>>>>>
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
//
// Arguments:
//   std::vector<ZVarHisto>& VecVarHisto: vector of objects to be filled
//   double w: weight
//   TLorentzVector* t: top quark momentum
//   TLorentzVector* tbar: antitop quark momentum
//   TLorentzVector* vecLepM: leptoni momentum (if needed, can be omitted)
//   TLorentzVector* vecLepP: lepton+ momentum (if needed, can be omitted)
//
void FillHistos(std::vector<ZVarHisto>& VecVarHisto, double w, TLorentzVector* t, TLorentzVector* tbar, TLorentzVector* vecLepM = NULL, TLorentzVector* vecLepP = NULL)
{
  // momentum of ttbar pair
  TLorentzVector ttbar = *t + *tbar;
  // loop over provided histograms to be filled
  for(int h = 0; h < VecVarHisto.size(); h++)
  {
    // retrieve variable name
    TString var = VecVarHisto[h].V();
    // retrieve histogram
    TH1* histo = VecVarHisto[h].H();
    // now fill histograms depending on the variable:
    // top pT
    //std::cout<<var<<std::endl;
    if(var == "ptt") 
      histo->Fill(t->Pt(), w);
    // antitop pT
    else if(var == "ptat") 
      histo->Fill(tbar->Pt(), w);
    // top pT, antitop pT (two entries per one event)
    else if(var == "pttat") 
    {
      histo->Fill(t->Pt(), w);
      histo->Fill(tbar->Pt(), w);
    }
    // ttbar pT
    else if(var == "pttt") 
      histo->Fill(ttbar.Pt(), w);
    // top rapidity
    else if(var == "yt") 
      histo->Fill(t->Rapidity(), w);
    // antitop pT
    else if(var == "yat") 
      histo->Fill(tbar->Rapidity(), w);
    // top rapidity, antitop rapidity (two entries per one event)
    else if(var == "ytat") 
    {
      histo->Fill(t->Rapidity(), w);
      histo->Fill(tbar->Rapidity(), w);
    }
    // ttbar rapidity
    else if(var == "ytt") 
      histo->Fill(ttbar.Rapidity(), w);
    // ttbar invariant mass
    else if(var == "mtt") 
      histo->Fill(ttbar.M(), w);
    else if(var == "phitt") 
      histo->Fill(t->Phi(), w);
    else if(var == "dphitt") 
    {
    double dphi = fabs(t->Phi() - tbar->Phi());
    if (dphi > M_PI) 
        dphi = 2 * M_PI - dphi;
    histo->Fill(dphi, w);
    }
    else if(var == "ptl") 
    {
      histo->Fill(vecLepM->Pt(), w);
      histo->Fill(vecLepP->Pt(), w);
    }
    // uknown (not implemented) variable
    // (you can implement more variables here if needed)
    else
    {
      //printf("Error: unknown variable %s\n", var.Data());
      //exit(1);
      continue;
    } // end variable for histo
  } // end loop over histos
}
void FillHistos_lkr(std::vector<ZVarHisto>& VecVarHisto, double w, TLorentzVector* ttbar, TLorentzVector* vecLepM = NULL, TLorentzVector* vecLepP = NULL)
{
  // loop over provided histograms to be filled
  for(int h = 0; h < VecVarHisto.size(); h++)
  {
    // retrieve variable name
    TString var = VecVarHisto[h].V();
    // retrieve histogram
    TH1* histo = VecVarHisto[h].H();
    // now fill histograms depending on the variable:
    // ttbar pT
    if(var == "pttt") 
      histo->Fill(ttbar->Pt(), w);
    // ttbar rapidity
    else if(var == "ytt") 
      histo->Fill(ttbar->Rapidity(), w);
    // ttbar invariant mass
    else if(var == "mtt") 
      histo->Fill(ttbar->M(), w);
    // Azimuthal angle
    else if(var == "phitt") 
      histo->Fill(ttbar->Phi(), w);
    else if(var == "phitt") 
      histo->Fill(ttbar->Phi(), w);
    // lepton pT
    else if(var == "ptl") 
    {
      histo->Fill(vecLepM->Pt(), w);
      histo->Fill(vecLepP->Pt(), w);
    }
    // uknown (not implemented) variable
    // (you can implement more variables here if needed)
    else
    {
      //printf("Error: unknown variable %s\n", var.Data());
      //exit(1);
      continue;
    } // end variable for histo
  } // end loop over histos
}
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
// >>>>>>>>>> Store hostogram from ZVarHisto class >>>>>>>>>>>>>>>>>>>>>
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
//
// Store bunch of histogram (argument std::vector<ZVarHisto>& VecVarHisto)
// (stores a copy of histogram)
//
void StoreHistos(std::vector<ZVarHisto>& VecVarHisto)
{
  for(int h = 0; h < VecVarHisto.size(); h++)
  {
    TH1* histo = VecVarHisto[h].H();
    TString name = histo->GetName();
    TString title = histo->GetTitle();
    histo->SetNameTitle(name, title);
    histo->Write();
  }
}
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
// >>>>>> Input parameters for eventreco routine (see below)  >>>>>>>>>>
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
class ZEventRecoInput
{
  public:
    TString Name; // name pattern (to be used in output histograms)
    std::vector<ZVarHisto> VecVarHisto; // container with needed histograms
    int Channel; // 1 ee, 2 mumu, 3 emu
    int Type; // 1 data, 2 MC signal, 3 MC ttbar other, 4 MC background
    bool Gen; // if true, the histogram is filled at true level
    std::vector<TString> VecInFile; // container with input files
    double Weight; // weight for histogram filling
    std::string nameConfigFile; // option config file
    bool StoreAllVars; // apply to signal MC, write out all events with all needed input variables + kine reco output
    
    // contstructor
    ZEventRecoInput()
    {
      // set default values
      Weight = 1.0;
      Gen = false;
      StoreAllVars = false;
    }
    
    // add one more input file (str) to the chain
    void AddToChain(const TString& str)
    {
      VecInFile.push_back(str);
    }

    // erase all input files form the chain
    void ClearChain()
    {
      VecInFile.clear();
    }
};

// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
// >>>>>>>>> Basic routine for ttbar event reconstruction >>>>>>>>>>>>>>
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
void eventreco(ZEventRecoInput in)
{ 
  printf("****** EVENTRECO ******\n");
  printf("input sample: %s\n", in.Name.Data());
  printf("type: %d   channel: %d\n", in.Type, in.Channel);
  printf("StoreAllVars: %d\n", in.StoreAllVars);
  
  // steering
  // b-tagging discriminator for Combined Secondary Vertex Loose 
  // (consult https://twiki.cern.ch/twiki/bin/view/CMSPublic/BtagRecommendation2011OpenData)
  const double bTagDiscrL = 0.244; 
  // directory for output ROOT files with histograms
  TString outDir = gHistDir; 

  // this flag determines whether generator level information is available
  // (should be available for signal MC)
  bool flagMC = (in.Type == 2 || in.Type == 3);
  if(in.StoreAllVars && (in.Type != 2)) {
    printf("ERROR: inconsistent StoreAllVars = %d and Type = %d\n", in.StoreAllVars, in.Type);
    exit(1);
  }
  
  // output file
  TFile* fout = TFile::Open(TString::Format("%s/%s-c%d.root", outDir.Data(), in.Name.Data(), in.Channel), "recreate");
  
  // input tree
  TChain* chain = new TChain("tree");
  for(int f = 0; f < in.VecInFile.size(); f++)
    chain->Add(in.VecInFile[f]);
  ZTree* preselTree = new ZTree(flagMC, in.StoreAllVars);
  preselTree->Init(chain);

  // process generator level, if needed
  if(in.Gen)
  {
    chain->SetBranchStatus("*", 0);
    chain->SetBranchStatus("mcEventType", 1);
    chain->SetBranchStatus("mcT", 1);
    chain->SetBranchStatus("mcTbar", 1);
  }
    
  // event counters
  long nSel = 0;
  long nReco = 0;
  long nGen = 0;
  
  // histograms for kinematic reconstruction debugging
  // (not needed in physics analysis, not stored)
  //TH1D* hInacc = new TH1D("hInacc", "KinReco inaccuracy", 1000, 0.0, 100.0);
  //TH1D* hAmbig = new TH1D("hAmbig", "KinReco ambiguity", 100, 0.0, 100.0);

  // vector of kinematic reconstruction methods
  std::vector<KinRecoBase*> kinrecos;
  // vector of kinematic reconstruction methods on generator level
  // (using different names in order to store their output in separate branches)
  std::vector<KinRecoBase*> kinrecos_gen;
  auto add_kr_to_vec = [&kinrecos_gen](KinRecoBase* kr, const std::string& name, std::vector<KinRecoBase*>& vec) {
    kr->SetName(name);
    vec.push_back(kr);
  };
  if (read_int(in.nameConfigFile, "kr_FKR", 1)) {
    kinrecos.push_back(new FKR());
    add_kr_to_vec(new FKR(), "fkr_gen", kinrecos_gen);
  }
  if (read_int(in.nameConfigFile, "kr_SKR", 1)) {
    kinrecos.push_back(new SKR());
    add_kr_to_vec(new SKR(), "skr_gen", kinrecos_gen);
  }
  LKR* kinreco_lkr = new LKR(); // store this pointer in order to call SelectbestJets()
  if (read_int(in.nameConfigFile, "kr_LKR", 1)) {
    kinrecos.push_back(kinreco_lkr);
    add_kr_to_vec(new LKR(), "lkr_gen", kinrecos_gen);
  }
  if (read_int(in.nameConfigFile, "kr_LKRv2", 1)) {
    kinrecos.push_back(new LKRv2());
    add_kr_to_vec(new LKRv2(), "lkrv2_gen", kinrecos_gen);
  }
  if (read_int(in.nameConfigFile, "kr_LKRv3", 1)) {
    kinrecos.push_back(new LKRv3());
    add_kr_to_vec(new LKRv3(), "lkrv3_gen", kinrecos_gen);
  }
  // LKRnn: нейромережева корекція поверх LKRv3.
  //   krNNGen=0 -> детекторна гілка бере ДЕТЕКТОРНУ модель (звичайний режим);
  //   krNNGen=1 -> детекторна гілка бере ГЕНЕРАТОРНУ модель (крос-тест gen-моделі на det-даних).
  // Генераторна гілка (lkrnn_gen) ЗАВЖДИ використовує генераторну модель — застосовувати
  // детекторну модель на gen-рівні сенсу не має.
  int krNNGen = read_int(in.nameConfigFile, "krNNGen", 0);
  if (read_int(in.nameConfigFile, "kr_LKRnn", 1)) {
    printf("[I] LKRnn: детекторна гілка -> %s модель\n", krNNGen ? "ГЕНЕРАТОРНА (крос-тест)" : "детекторна");
    kinrecos.push_back(new LKRnn(krNNGen));
    add_kr_to_vec(new LKRnn(true), "lkrnn_gen", kinrecos_gen);
  }
  // окремий LKRv3 для дампу 26 NN-ознак (LKRv3::computeFeatures) — незалежно від kr_LKRv3
  LKRv3* kinreco_lkrv3_feat = new LKRv3();
  // vector of variables for kinematic reconstruction
  std::vector<KRVAR*> krvars;
  if (read_int(in.nameConfigFile, "krvar_mtt", 1)) krvars.push_back(new Mtt());
  if (read_int(in.nameConfigFile, "krvar_ytt", 1)) krvars.push_back(new Ytt());
  if (read_int(in.nameConfigFile, "krvar_pttt", 1)) krvars.push_back(new Pttt());
  if (read_int(in.nameConfigFile, "krvar_phitt", 1)) krvars.push_back(new Phitt());
  if (read_int(in.nameConfigFile, "krvar_dphitt", 1)) krvars.push_back(new Dphitt());

  // determine number of events
  long nEvents = chain->GetEntries();
  // read maximum number of events from config file (-1 for no limit)
  int maxNEvents = read_int(in.nameConfigFile, "maxNEvents", -1);
  //limit it if exceeds the specified maximum number
  if(maxNEvents >=0 && nEvents > maxNEvents)
    nEvents = maxNEvents;
  printf("nEvents: %ld\n", nEvents);
  TFile *outputFile = nullptr;
  // TTree to store kinematic reconstruction output
  TTree *tree_kr = nullptr;
  if(in.Name == "mcSigReco") {
    std::string suffix = in.StoreAllVars ? "_full" : "";
    outputFile = new TFile(TString::Format("ttbar_output%s_%d.root", suffix.c_str(), in.Channel), "RECREATE");
    tree_kr = new TTree("ttbarTree", "Tree storing ttbar event variables");
    for (auto& kr : kinrecos) {
      kr->init(tree_kr, krvars);
    }
    for (auto& kr : kinrecos_gen) {
      kr->init(tree_kr, krvars);
    }
  }
  float mtt_gen, ytt_gen, pttt_gen, phitt_gen, dphitt_gen;
  int reco_passed_selection, reco_passed_selectbestjets;
  float mcLm[4], mcLp[4], mcJ1[4], mcJ2[4];
  float mcMetPx, mcMetPy;
  float recoLm[4], recoLp[4], recoJ1[4], recoJ2[4];
  float recoMetPx, recoMetPy;
  float nn_features[26], nn_features_gen[26]; // 26 NN-ознак (det / gen), -999 якщо джети не відібрані
  if (tree_kr) {
    //gen branches
    tree_kr->Branch("mtt_gen", &mtt_gen, "mtt_gen/F");
    tree_kr->Branch("phitt_gen", &phitt_gen, "phitt_gen/F");
    tree_kr->Branch("ytt_gen", &ytt_gen, "ytt_gen/F");
    tree_kr->Branch("pttt_gen", &pttt_gen, "pttt_gen/F");
    tree_kr->Branch("dphitt_gen", &dphitt_gen, "dphitt_gen/F");
    if (in.StoreAllVars) {
      tree_kr->Branch("reco_passed_selection", &reco_passed_selection, "reco_passed_selection/I");
      tree_kr->Branch("reco_passed_selectbestjets", &reco_passed_selectbestjets, "reco_passed_selectbestjets/I");
      tree_kr->Branch("mcLp", &mcLp, "mcLp[4]/F");
      tree_kr->Branch("mcLm", &mcLm, "mcLm[4]/F");
      tree_kr->Branch("mcJ1", &mcJ1, "mcJ1[4]/F");
      tree_kr->Branch("mcJ2", &mcJ2, "mcJ2[4]/F");
      tree_kr->Branch("mcMetPx", &mcMetPx, "mcMetPx/F");
      tree_kr->Branch("mcMetPy", &mcMetPy, "mcMetPy/F");
      tree_kr->Branch("recoLp", &recoLp, "recoLp[4]/F");
      tree_kr->Branch("recoLm", &recoLm, "recoLm[4]/F");
      tree_kr->Branch("recoJ1", &recoJ1, "recoJ1[4]/F");
      tree_kr->Branch("recoJ2", &recoJ2, "recoJ2[4]/F");
      tree_kr->Branch("recoMetPx", &recoMetPx, "recoMetPx/F");
      tree_kr->Branch("recoMetPy", &recoMetPy, "recoMetPy/F");
      tree_kr->Branch("nn_features", nn_features, "nn_features[26]/F");
      tree_kr->Branch("nn_features_gen", nn_features_gen, "nn_features_gen[26]/F");
    }
  }
  // event loop
  for(int e = 0; e < nEvents; e++)
  {
    if(e%100000 == 0) {
      printf("done %d events [%.0f%%]\n", e, 100.*e/nEvents);
    }
    chain->GetEntry(e);
    if(flagMC)
    {
      // skip background events for MC signal
      if(in.Type == 2 && preselTree->mcEventType != in.Channel) continue;
      // skip signal events for MC 'ttbar other' (background)
      if(in.Type == 3 && preselTree->mcEventType == in.Channel) continue;
    }
    // process generator level if needed
    if(in.Gen)
    {
      // prepare four vectors for top and antitop
      TLorentzVector t, tbar;
      t.SetXYZM(preselTree->mcT[0], preselTree->mcT[1], preselTree->mcT[2], preselTree->mcT[3]);
      tbar.SetXYZM(preselTree->mcTbar[0], preselTree->mcTbar[1], preselTree->mcTbar[2], preselTree->mcTbar[3]);
      // fill histos
      double w = in.Weight;
      FillHistos(in.VecVarHisto, w, &t, &tbar);
      nGen++;
      continue;
    }
    if(in.Type > 1)
      nGen++;
    
    auto process_reco_level = [bTagDiscrL, &in, preselTree](TLorentzVector& vecLepM, TLorentzVector& vecLepP, std::vector<TLorentzVector>& vecJets) {
      // primary vertex selection
      if(preselTree->Npv < 1 || preselTree->pvNDOF < 4 || preselTree->pvRho > 2.0 || TMath::Abs(preselTree->pvZ) > 24.0)
        return false;
      // primary dataset name
      //TString inFile = chain->GetCurrentFile()->GetName();
      // select dilepton pair
      double maxPtDiLep = -1.0; // initialise with a negative value to determine later on whether a dilepton pair is found in the event
      bool trig = false;
      // *****************************************
      // ***************** emu *******************
      // *****************************************
      if(in.Channel == 3)
      {
        // trigger: 12th to 17th bits
        // (accept the event if at least one needed trigger bit is fired) 
        for(int bit = 12; bit < 17; bit++)
          if((preselTree->Triggers >> bit) & 1)
          {
            trig = true;
            break;
          }
        // call dileption selection routine (see selection.h for description)
        if(trig)
          SelectDilepEMu(preselTree, vecLepM, vecLepP, maxPtDiLep);
      }
      // *****************************************
      // ***************** ee ********************
      // *****************************************
      if(in.Channel == 1)
      {
        // trigger: 6th to 11th bits
        for(int bit = 6; bit < 11; bit++)
          if((preselTree->Triggers >> bit) & 1)
          {
            trig = true;
            break;
          }
        double met = TMath::Sqrt(TMath::Power(preselTree->metPx, 2.0) + TMath::Power(preselTree->metPy, 2.0));
        // additinal requirement on the missing transverse energy
        if(trig && met > 30.0)
          SelectDilepEE(preselTree, vecLepM, vecLepP, maxPtDiLep);
      }
      // *****************************************
      // **************** mumu *******************
      // *****************************************
      if(in.Channel == 2)
      {
        // trigger: 0th to 5th bits
        for(int bit = 0; bit < 5; bit++)
          if((preselTree->Triggers >> bit) & 1)
          {
            trig = true;
            break;
          }
        double met = TMath::Sqrt(TMath::Power(preselTree->metPx, 2.0) + TMath::Power(preselTree->metPy, 2.0));
        // additinal requirement on the missing transverse energy
        if(trig && met > 30.0)
          SelectDilepMuMu(preselTree, vecLepM, vecLepP, maxPtDiLep);
      }
      // check if there is a dilepton pair found, otherwise skip the event
      if(maxPtDiLep < 0.0)
        return false;
      // dilepton pair found, now select jets; 
      // all jets are stored for kinematic reconstruction
      bool oneBTagJet = false;
      for(int j = 0; j < preselTree->Njet; j++)
      {
        if(TMath::Abs(preselTree->jetEta[j]) > 2.4)
          continue;
        TLorentzVector vecJet;
        vecJet.SetPtEtaPhiM(preselTree->jetPt[j], preselTree->jetEta[j], preselTree->jetPhi[j], preselTree->jetMass[j]);
        // subtract muon and electron energy fractions
        double corrE = vecJet.E() - preselTree->jetMuEn[j] - preselTree->jetElEn[j];
        double corrPt = preselTree->jetPt[j] * corrE / vecJet.E();\
        // require pT(jet) > 30 GeV
        if(corrPt < 30.0)
          continue;
        TLorentzVector corrVec;
        corrVec.SetPtEtaPhiE(corrPt, preselTree->jetEta[j], preselTree->jetPhi[j], corrE);
        // b-tagging: check if there at least one b-tagged jet
        // for b-tagged jet make the jet mass negative: this is for proper 
        // identification of b-tagged jets in the kinematic reconstruction
        if(preselTree->jetBTagDiscr[j] > bTagDiscrL)
        {
          corrVec.SetPtEtaPhiM(corrVec.Pt(), corrVec.Eta(), corrVec.Phi(), -1 * corrVec.M());
          oneBTagJet = true;
        }
        vecJets.push_back(corrVec);
      }
      // if there are no two jets, skip the event
      if(vecJets.size() < 2)
        return false;
      // require at least one b-tagged jet
      if(!oneBTagJet)
        return false;
      return true;
    };
    
    // process reco level if needed
    TLorentzVector vecLepM, vecLepP;
    std::vector<TLorentzVector> vecJets;
    reco_passed_selection = process_reco_level(vecLepM, vecLepP, vecJets);
    reco_passed_selectbestjets = 0;
    //
    if (reco_passed_selection == 0) {
      if(in.StoreAllVars == 0) 
        continue;
      else {
        for (auto& kr : kinrecos) {
          kr->reset_vars();
          for(int i = 0; i < 4; i++) {
            recoLm[i] = recoLp[i] = recoJ1[i] = recoJ2[i] = -999.;
          }
          recoMetPx = recoMetPy = -999.;
        }
        for(int i = 0; i < 26; i++) nn_features[i] = -999.;
      }
    }
    else {
      if(in.StoreAllVars == 1) {
        recoLm[0] = vecLepM.Px();
        recoLm[1] = vecLepM.Py();
        recoLm[2] = vecLepM.Pz();
        recoLm[3] = vecLepM.M();
        recoLp[0] = vecLepP.Px();
        recoLp[1] = vecLepP.Py();
        recoLp[2] = vecLepP.Pz();
        recoLp[3] = vecLepP.M();
        TLorentzVector jetBest1, jetBest2;
        reco_passed_selectbestjets = kinreco_lkr->selectBestJetsPublic(vecLepM, vecLepP, vecJets, preselTree->jetBTagDiscr, bTagDiscrL, jetBest1, jetBest2);
        if(reco_passed_selectbestjets) {
          recoJ1[0] = jetBest1.Px();
          recoJ1[1] = jetBest1.Py();
          recoJ1[2] = jetBest1.Pz();
          recoJ1[3] = jetBest1.M();
          recoJ2[0] = jetBest2.Px();
          recoJ2[1] = jetBest2.Py();
          recoJ2[2] = jetBest2.Pz();
          recoJ2[3] = jetBest2.M();
        }
        else {
          for(int i = 0; i < 4; i++) {
            recoJ1[i] = recoJ2[i] = -999.;
          }
        }
        recoMetPx = preselTree->metPx;
        recoMetPy = preselTree->metPy;
        // 26 NN-ознак (det): з ВЛАСНИХ LKRv3-джетів (не LKR-них recoJ1/recoJ2),
        // щоб збігатися з тим, що LKRnn рахує на inference
        for(int i = 0; i < 26; i++) nn_features[i] = -999.;
        TLorentzVector jbf1, jbf2;
        if(kinreco_lkrv3_feat->selectBestJetsPublic(vecLepM, vecLepP, vecJets, preselTree->jetBTagDiscr, bTagDiscrL, jbf1, jbf2)) {
          std::vector<float> ftr = kinreco_lkrv3_feat->computeFeatures(vecLepM, vecLepP, jbf1, jbf2, preselTree->metPx, preselTree->metPy);
          for(int i = 0; i < 26; i++) nn_features[i] = ftr[i];
        }
      }
    }
    // event selection done: increment the counter of selected events
    if(reco_passed_selection)
      nSel++;
    
    // fill generator level top, tbar and ttbar variables
    if(tree_kr) {
      TLorentzVector t_gen, tbar_gen;
      t_gen.SetXYZM(preselTree->mcT[0], preselTree->mcT[1], preselTree->mcT[2], preselTree->mcT[3]);
      tbar_gen.SetXYZM(preselTree->mcTbar[0], preselTree->mcTbar[1], preselTree->mcTbar[2], preselTree->mcTbar[3]);
      mtt_gen=(t_gen+tbar_gen).M();
      ytt_gen=(t_gen+tbar_gen).Rapidity();
      pttt_gen=(t_gen+tbar_gen).Pt();
      phitt_gen=(t_gen+tbar_gen).Phi();
      double dphi_gen = fabs(t_gen.Phi() - tbar_gen.Phi());
      if (dphi_gen > M_PI) dphi_gen = 2 * M_PI - dphi_gen;
      dphitt_gen = dphi_gen;
      // fill other MC generator-level varaibles if needed
      if(in.StoreAllVars) {
        for(int i = 0; i < 4; i++) {
          mcLp[i] = preselTree->mcLp[i];
          mcLm[i] = preselTree->mcLm[i];
          mcJ1[i] = preselTree->mcB[i];
          mcJ2[i] = preselTree->mcBbar[i];
        }
        mcMetPx = preselTree->mcNu[0] + preselTree->mcNubar[0];
        mcMetPy = preselTree->mcNu[1] + preselTree->mcNubar[1];
        // run kinematic reconstruction on generator level
        auto make_vector = [](const float* p1, const float* p2 = nullptr) {
          std::vector<TLorentzVector> vec((p2 == nullptr) ? 1 : 2);
          TLorentzVector part;
          part.SetXYZM(p1[0], p1[1], p1[2], p1[3]);
          vec[0] = part;
          if(p2) {
            part.SetXYZM(p2[0], p2[1], p2[2], p2[3]);
            vec[1] = part;
          }
          return vec;
        };
        std::vector<TLorentzVector> vecLepM_gen = make_vector(mcLm);
        std::vector<TLorentzVector> vecLepP_gen = make_vector(mcLp);
        std::vector<TLorentzVector> vecJets_gen = make_vector(mcJ1, mcJ2);
        std::vector<float> fake_btag = {1., 1.};
        for (auto& kr : kinrecos_gen) {
          kr->reset_vars();
          std::vector<TLorentzVector> solution = kr->reconstruct(vecLepM_gen[0], vecLepP_gen[0], vecJets_gen, &fake_btag[0], bTagDiscrL, mcMetPx, mcMetPy);
          if(solution.size()) {
            kr->calculate_vars(solution[0], solution[1], solution[2]);
          }
        }
        // 26 NN-ознак (gen) з LKRv3-джетів на генераторних входах
        for(int i = 0; i < 26; i++) nn_features_gen[i] = -999.;
        TLorentzVector jbg1, jbg2;
        if(kinreco_lkrv3_feat->selectBestJetsPublic(vecLepM_gen[0], vecLepP_gen[0], vecJets_gen, &fake_btag[0], bTagDiscrL, jbg1, jbg2)) {
          std::vector<float> ftr = kinreco_lkrv3_feat->computeFeatures(vecLepM_gen[0], vecLepP_gen[0], jbg1, jbg2, mcMetPx, mcMetPy);
          for(int i = 0; i < 26; i++) nn_features_gen[i] = ftr[i];
        }
      }
    }

    // run kinematic reconstruction to restore the top and antitop momenta
    bool flagPassedKinRec = false; // status used to go further to fill histograms
    TLorentzVector t, tbar; // vectors used to fill histograms
    if (reco_passed_selection) {
      for (auto& kr : kinrecos) {
        kr->reset_vars();
        std::vector<TLorentzVector> solution = kr->reconstruct(vecLepM, vecLepP, vecJets, preselTree->jetBTagDiscr, bTagDiscrL, preselTree->metPx, preselTree->metPy);
        if(solution.size()) {
          kr->calculate_vars(solution[0], solution[1], solution[2]);
        }
        // TEMPORARY use SKR solution to fill histograms
        if (kr->GetName() == "skr") {
          if(solution.size()) {
            flagPassedKinRec = true;
            t = solution[0];
            tbar = solution[1];
          }
        }
      }
      if(flagPassedKinRec>0)// successfull skr
      {
        // print the top and antitop momenta, if needed
        //printf("top:      (%8.3f  %8.3f  %8.3f  %8.3f)\n", t.X(), t.Y(), t.Z(), t.M());
        //printf("antitop:  (%8.3f  %8.3f  %8.3f  %8.3f)\n", tbar.X(), tbar.Y(), tbar.Z(), tbar.M());
        nReco++;
        
        // fill histograms
        double w = in.Weight;
        //std::cout<<phitt_skr<<std::endl;
        //FillHistos_lkr(in.VecVarHisto, w, &ttbar_lkr, &vecLepM, &vecLepP);
        FillHistos(in.VecVarHisto, w, &t, &tbar, &vecLepM, &vecLepP);
      } // end kinreco
    }
    if (tree_kr) {
      tree_kr->Fill();
    }
  } // end event loop
  
  // print the numbers of selected events and events with successfull kinematic reconstruction
  printf("nSel  : %ld\n", nSel);
  printf("nReco : %ld\n", nReco);
  // for signal MC, print the number of signal events at generator level and detector efficiency
  // (with and without kinematic reconstruction)
  if(in.Type == 2) 
  {
    printf("nGen  : %ld\n", nGen);
    printf("C = %.2f%% (no KINRECO %.2f%%)\n", 100. * nReco / nGen, 100. * nSel / nGen);
  }

  // store histograms, close output file
  fout->cd();
  StoreHistos(in.VecVarHisto);
  fout->Close();
  if (tree_kr) {
    outputFile->cd();
    tree_kr->Write();
    outputFile->Close();
  }
  for (auto& kr : kinrecos) {
    delete kr;
  }
  for (auto& kr : kinrecos_gen) {
    delete kr;
  }
  for (auto& krvar : krvars) {
    delete krvar;
  }
}

#endif
