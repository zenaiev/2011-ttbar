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
      histo->Fill(ttbar.Phi(), w);
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
    
    // contstructor
    ZEventRecoInput()
    {
      // set default values
      Weight = 1.0;
      Gen = false;
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
  
  // steering
  // b-tagging discriminator for Combined Secondary Vertex Loose 
  // (consult https://twiki.cern.ch/twiki/bin/view/CMSPublic/BtagRecommendation2011OpenData)
  const double bTagDiscrL = 0.244; 
  // directory for output ROOT files with histograms
  TString outDir = gHistDir; 

  // this flag determines whether generator level information is available
  // (should be available for signal MC)
  bool flagMC = (in.Type == 2 || in.Type == 3);
  
  // output file
  TFile* fout = TFile::Open(TString::Format("%s/%s-c%d.root", outDir.Data(), in.Name.Data(), in.Channel), "recreate");
  
  // input tree
  TChain* chain = new TChain("tree");
  for(int f = 0; f < in.VecInFile.size(); f++)
    chain->Add(in.VecInFile[f]);
  ZTree* preselTree = new ZTree(flagMC);
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
  if (read_int(in.nameConfigFile, "kr_FKR", 0)) kinrecos.push_back(new FKR());
  if (read_int(in.nameConfigFile, "kr_SKR", 1)) kinrecos.push_back(new SKR());
  if (read_int(in.nameConfigFile, "kr_LKR", 0)) kinrecos.push_back(new LKR());

  // vector of variables for kinematic reconstruction
  std::vector<KRVAR*> krvars;
  if (read_int(in.nameConfigFile, "krvar_mtt", 1)) krvars.push_back(new Mtt());
  if (read_int(in.nameConfigFile, "krvar_ytt", 1)) krvars.push_back(new Ytt());
  if (read_int(in.nameConfigFile, "krvar_pttt", 1)) krvars.push_back(new Pttt());
  if (read_int(in.nameConfigFile, "krvar_phitt", 1)) krvars.push_back(new Phitt());

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
    outputFile = new TFile(TString::Format("ttbar_output_%d.root", in.Channel), "RECREATE");
    tree_kr = new TTree("ttbarTree", "Tree storing ttbar event variables");
    for (auto& kr : kinrecos) {
      kr->init(tree_kr, krvars);
    }
  }
  float mtt_gen, ytt_gen, pttt_gen, phitt_gen;
  if (tree_kr) {
    //gen branches
    tree_kr->Branch("mtt_gen", &mtt_gen, "mtt_gen/F");
    tree_kr->Branch("phitt_gen", &phitt_gen, "phitt_gen/F");
    tree_kr->Branch("ytt_gen", &ytt_gen, "ytt_gen/F");
    tree_kr->Branch("pttt_gen", &pttt_gen, "pttt_gen/F");
  }
  // event loop
  for(int e = 0; e < nEvents; e++)
  {
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
    
    // process reco level if needed
    //
    // primary vertex selection
    if(preselTree->Npv < 1 || preselTree->pvNDOF < 4 || preselTree->pvRho > 2.0 || TMath::Abs(preselTree->pvZ) > 24.0)
      continue;
    // primary dataset name
    //TString inFile = chain->GetCurrentFile()->GetName();
    // select dilepton pair
    TLorentzVector vecLepM, vecLepP;
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
      continue;
    // dilepton pair found, now select jets; 
    // all jets are stored for kinematic reconstruction
    std::vector<TLorentzVector> vecJets;
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
      continue;
    // require at least one b-tagged jet
    if(!oneBTagJet)
      continue;
    // event selection done: increment the counter of selected events
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
    }

    // run kinematic reconstruction to restore the top and antitop momenta
    bool flagPassedKinRec = false; // status used to go further to fill histograms
    TLorentzVector t, tbar; // vectors used to fill histograms
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
  for (auto& krvar : krvars) {
    delete krvar;
  }
}

#endif
