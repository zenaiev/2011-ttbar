// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
// This class contains all needed variables to read ROOT ntuples for ttbar analysis
// (automaticlly produced by ROOT, then slightly tuned manually)
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

#ifndef ZTree_h
#define ZTree_h

#include <TROOT.h>
#include <TChain.h>

// Class which gives access to all information in each event stored in ntuples
class ZTree {
public :
   // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
   // public members (for direct access outside the class)
   // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    
   // pointer to the analyzed TTree or TChain
   TTree *fChain;  

   // MC flag (true for MC, false for data)
   bool _flagMC;

   // write out all events with all needed input variables + kine reco output
   int _storeAllVars;

   // variable array max sizes
   static const int maxNel = 10; // electrons
   static const int maxNmu = 10; // muons
   static const int maxNjet = 25; // jets

   // Ntuple variables (their description can be found also in Analyzer/src/Analyzer.cc)
   //[N] means that this is fixed size array with N elements
   Int_t           evRunNumber; // run number
   Int_t           evEventNumber; // event number
   
   Int_t           Nmu; // number of muons
   Float_t         muPt[maxNmu];   //[Nmu] muon pT
   Float_t         muEta[maxNmu];   //[Nmu] muon eta
   Float_t         muPhi[maxNmu];   //[Nmu] muon phi
   Float_t         muIso03[maxNmu];   //[Nmu] muon isolation delta_R=0.3
   Float_t         muIso04[maxNmu];   //[Nmu] muon isolation delta_R=0.4
   Int_t           muHitsValid[maxNmu];   //[Nmu] muon valid hits number
   Int_t           muHitsPixel[maxNmu];   //[Nmu] muon pixel hits number
   Float_t         muDistPV0[maxNmu];   //[Nmu] muon distance to the primary vertex (projection on transverse plane)
   Float_t         muDistPVz[maxNmu];   //[Nmu] muon distance to the primary vertex (z projection)
   Float_t         muTrackChi2NDOF[maxNmu];   //[Nmu] muon track number of degrees of freedom

   Int_t           Nel; // number of electrons
   Float_t         elPt[maxNel];   //[Nel] electron pT
   Float_t         elEta[maxNel];   //[Nel] electron eta
   Float_t         elPhi[maxNel];   //[Nel] electron phi
   Float_t         elIso03[maxNel];   //[Nel] electron isolation delta_R=0.3
   Float_t         elIso04[maxNel];   //[Nel] electron isolation delta_R=0.4
   Int_t           elConvFlag[maxNel];   //[Nel] (not used) electron conversion flag
   Float_t         elConvDist[maxNel];   //[Nel] (not used) electron conversion distance
   Float_t         elConvDcot[maxNel];   //[Nel] (not used) electron conversion cotangent
   Float_t         elMissHits[maxNel];   //[Nel] electron missing hits number

   Int_t           Njet; // number of jets
   Float_t         jetPt[maxNjet];   //[Njet] jet pT
   Float_t         jetEta[maxNjet];   //[Njet] jet eta
   Float_t         jetPhi[maxNjet];   //[Njet] jet phi
   Float_t         jetMass[maxNjet];   //[Njet] jet mass
   Float_t         jetMuEn[maxNjet];   //[Njet] jet muon energy
   Float_t         jetElEn[maxNjet];   //[Njet] jet electron energy
   Float_t         jetBTagDiscr[maxNjet];   //[Njet] jet b-tagging discriminant (Combined Secondary Vertex, CSV)
   Float_t         jetBTagMatchDiff1[maxNjet];   //[Njet] (not used, for checks) jet b-tagging: eta-phi distance to the closest matched jet
   Float_t         jetBTagMatchDiff2[maxNjet];   //[Njet] (not used, for checks) jet b-tagging: eta-phi distance to the second closest matched jet
   Float_t         metPx; // missing transverse energy x component
   Float_t         metPy; // missing transverse energy y component
   Int_t           Npv; // total number of primary vertices
   Int_t           pvNDOF; // number of degrees of freedom of the primary vertex
   Float_t         pvZ; // z component of the primary vertex
   Float_t         pvRho; // rho of the primary vertex (projection on transverse plane)
   Int_t           Triggers; // trigger bits
   
   // variables for MC only
   Int_t           mcEventType; // type of event: 1 ttbar decay into ee, 2 ttbar decay into mumu, 3 ttbar decay into emu, 0 anything else
   float mcT[4];    // top quark four momentum
   float mcTbar[4]; // antitop quark four momentum
   float mcLp[4];    // l+ four momentum
   float mcLm[4];    // l- four momentum
   float mcB[4];    // b four momentum
   float mcBbar[4];    // anti-b four momentum
   float mcNu[4]; // nu four momentum
   float mcNubar[4]; // anti-nu four momentum

   // List of branches (their names follow variable names with prefix b_)
   TBranch        *b_evRunNumber;   //!
   TBranch        *b_evEventNumber;   //!
   TBranch        *b_Nmu;   //!
   TBranch        *b_muPt;   //!
   TBranch        *b_muEta;   //!
   TBranch        *b_muPhi;   //!
   TBranch        *b_muIso03;   //!
   TBranch        *b_muIso04;   //!
   TBranch        *b_muHitsValid;   //!
   TBranch        *b_muHitsPixel;   //!
   TBranch        *b_muDistPV0;   //!
   TBranch        *b_muDistPVz;   //!
   TBranch        *b_muTrackChi2NDOF;   //!
   TBranch        *b_Nel;   //!
   TBranch        *b_elPt;   //!
   TBranch        *b_elEta;   //!
   TBranch        *b_elPhi;   //!
   TBranch        *b_elIso03;   //!
   TBranch        *b_elIso04;   //!
   TBranch        *b_elConvFlag;   //!
   TBranch        *b_elConvDist;   //!
   TBranch        *b_elConvDcot;   //!
   TBranch        *b_elMissHits;   //!
   TBranch        *b_Njet;   //!
   TBranch        *b_jetPt;   //!
   TBranch        *b_jetEta;   //!
   TBranch        *b_jetPhi;   //!
   TBranch        *b_jetMass;   //!
   TBranch        *b_jetMuEn;   //!
   TBranch        *b_jetElEn;   //!
   TBranch        *b_jetBTagDiscr;   //!
   TBranch        *b_jetBTagMatchDiff1;   //!
   TBranch        *b_jetBTagMatchDiff2;   //!
   TBranch        *b_metPx;   //!
   TBranch        *b_metPy;   //!
   TBranch        *b_Npv;   //!
   TBranch        *b_pvNDOF;   //!
   TBranch        *b_pvZ;   //!
   TBranch        *b_pvRho;   //!
   TBranch        *b_Triggers;   //!
   // for MC only
   TBranch        *b_mcEventType; //!
   TBranch        *b_mcT; //!
   TBranch        *b_mcTbar; //!
   TBranch        *b_mcLp; //!
   TBranch        *b_mcLm; //!
   TBranch        *b_mcB; //!
   TBranch        *b_mcBbar; //!
   TBranch        *b_mcNu; //!
   TBranch        *b_mcNubar; //!

   // constructor
   // argument: true for MC, false (default) for data
   ZTree(bool flagMC = false, bool storeAllVars = false) : fChain(0), _flagMC(flagMC), _storeAllVars(storeAllVars) { }
   
   // destructor
   virtual ~ZTree() { }
   
   // initialise with provided tree pointer
   virtual void    Init(TTree *tree);
};

// initialise with provided tree pointer
void ZTree::Init(TTree *tree)
{
   // Set branch addresses and branch pointers
   if (!tree) return;
   fChain = tree;
   fChain->SetMakeClass(1);

   fChain->SetBranchAddress("evRunNumber", &evRunNumber, &b_evRunNumber);
   fChain->SetBranchAddress("evEventNumber", &evEventNumber, &b_evEventNumber);
   fChain->SetBranchAddress("Nmu", &Nmu, &b_Nmu);
   fChain->SetBranchAddress("muPt", muPt, &b_muPt);
   fChain->SetBranchAddress("muEta", muEta, &b_muEta);
   fChain->SetBranchAddress("muPhi", muPhi, &b_muPhi);
   fChain->SetBranchAddress("muIso03", muIso03, &b_muIso03);
   fChain->SetBranchAddress("muIso04", muIso04, &b_muIso04);
   fChain->SetBranchAddress("muHitsValid", muHitsValid, &b_muHitsValid);
   fChain->SetBranchAddress("muHitsPixel", muHitsPixel, &b_muHitsPixel);
   fChain->SetBranchAddress("muDistPV0", muDistPV0, &b_muDistPV0);
   fChain->SetBranchAddress("muDistPVz", muDistPVz, &b_muDistPVz);
   fChain->SetBranchAddress("muTrackChi2NDOF", muTrackChi2NDOF, &b_muTrackChi2NDOF);
   fChain->SetBranchAddress("Nel", &Nel, &b_Nel);
   fChain->SetBranchAddress("elPt", elPt, &b_elPt);
   fChain->SetBranchAddress("elEta", elEta, &b_elEta);
   fChain->SetBranchAddress("elPhi", elPhi, &b_elPhi);
   fChain->SetBranchAddress("elIso03", elIso03, &b_elIso03);
   fChain->SetBranchAddress("elIso04", elIso04, &b_elIso04);
   fChain->SetBranchAddress("elConvFlag", elConvFlag, &b_elConvFlag);
   fChain->SetBranchAddress("elConvDist", elConvDist, &b_elConvDist);
   fChain->SetBranchAddress("elConvDcot", elConvDcot, &b_elConvDcot);
   fChain->SetBranchAddress("elMissHits", elMissHits, &b_elMissHits);
   fChain->SetBranchAddress("Njet", &Njet, &b_Njet);
   fChain->SetBranchAddress("jetPt", jetPt, &b_jetPt);
   fChain->SetBranchAddress("jetEta", jetEta, &b_jetEta);
   fChain->SetBranchAddress("jetPhi", jetPhi, &b_jetPhi);
   fChain->SetBranchAddress("jetMass", jetMass, &b_jetMass);
   fChain->SetBranchAddress("jetMuEn", jetMuEn, &b_jetMuEn);
   fChain->SetBranchAddress("jetElEn", jetElEn, &b_jetElEn);
   fChain->SetBranchAddress("jetBTagDiscr", jetBTagDiscr, &b_jetBTagDiscr);
   fChain->SetBranchAddress("jetBTagMatchDiff1", jetBTagMatchDiff1, &b_jetBTagMatchDiff1);
   fChain->SetBranchAddress("jetBTagMatchDiff2", jetBTagMatchDiff2, &b_jetBTagMatchDiff2);
   fChain->SetBranchAddress("metPx", &metPx, &b_metPx);
   fChain->SetBranchAddress("metPy", &metPy, &b_metPy);
   fChain->SetBranchAddress("Npv", &Npv, &b_Npv);
   fChain->SetBranchAddress("pvNDOF", &pvNDOF, &b_pvNDOF);
   fChain->SetBranchAddress("pvZ", &pvZ, &b_pvZ);
   fChain->SetBranchAddress("pvRho", &pvRho, &b_pvRho);
   fChain->SetBranchAddress("Triggers", &Triggers, &b_Triggers);
   // MC
   if(_flagMC) fChain->SetBranchAddress("mcEventType", &mcEventType, &b_mcEventType);
   if(_flagMC) fChain->SetBranchAddress("mcT", mcT, &b_mcT);
   if(_flagMC) fChain->SetBranchAddress("mcTbar", mcTbar, &b_mcTbar);
   if(_storeAllVars) {
      fChain->SetBranchAddress("mcLp", mcLp, &b_mcLp);
      fChain->SetBranchAddress("mcLm", mcLm, &b_mcLm);
      fChain->SetBranchAddress("mcB", mcB, &b_mcB);
      fChain->SetBranchAddress("mcBbar", mcBbar, &b_mcBbar);
      fChain->SetBranchAddress("mcNu", mcNu, &b_mcNu);
      fChain->SetBranchAddress("mcNubar", mcNubar, &b_mcNubar);
   }
}

#endif // #ifdef ZTree_h
