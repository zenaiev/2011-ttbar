#include <iostream>
#include <cstdlib>
#include <fstream>
#include <algorithm>
#include <sstream>

#include <TString.h>
#include <TSystem.h>
#include <TObjArray.h>
#include <Rtypes.h>

#include "sampleHelpers.h"






// --------------------- Functions defined in namespace Era -------------------------

Era::Era Era::convert(const TString& era)
{
    if(era == "run1_8tev") return run1_8tev;
    if(era == "run2_13tev_50ns") return run2_13tev_50ns;
    if(era == "run2_13tev_25ns") return run2_13tev_25ns;
    if(era == "run2_13tev_2015_25ns") return run2_13tev_2015_25ns;
    if(era == "run2_13tev_2016_25ns") return run2_13tev_2016_25ns;
    if(era == "run2_13tev_25ns_74X") return run2_13tev_25ns_74X;
    if(era == "") return undefined;
    std::cerr<<"Error in Era::convert()! Following conversion is not implemented: "<<era<<"\n...break\n"<<std::endl;
    exit(97);
}



TString Era::convert(const Era& era)
{
    if(era == run1_8tev) return "run1_8tev";
    if(era == run2_13tev_50ns) return "run2_13tev_50ns";
    if(era == run2_13tev_25ns) return "run2_13tev_25ns";
    if(era == run2_13tev_2015_25ns) return "run2_13tev_2015_25ns";
    if(era == run2_13tev_2016_25ns) return "run2_13tev_2016_25ns";
    if(era == run2_13tev_25ns_74X) return "run2_13tev_25ns_74X";
    if(era == undefined) return "";
    std::cerr<<"Error in Era::convert()! Conversion is not implemented\n...break\n"<<std::endl;
    exit(97);
}



double Era::energyInTev(const Era era)
{
    if(era == run1_8tev) return 8.;
    if(era == run2_13tev_50ns) return 13.;
    if(era == run2_13tev_25ns) return 13.;
    if(era == run2_13tev_2015_25ns) return 13.;
    if(era == run2_13tev_2016_25ns) return 13.;
    if(era == run2_13tev_25ns_74X) return 13.;
    if(era == undefined) return -1.;
    std::cerr<<"Error in Era::energyInTev()! No energy value defined for requested era\n...break\n"<<std::endl;
    exit(97);
}





// --------------------- Functions defined in namespace JetPileupId -------------------------

JetPileupId::WorkingPoint JetPileupId::convertWorkingPoint(const std::string& wp)
{
    if(wp == "none") return none;
    if(wp == "L") return L;
    if(wp == "M") return M;
    if(wp == "T") return T;
    if(wp == "") return undefinedWP;
    std::cerr<<"Error in JetPileupId::convertWorkingPoint()! Following conversion is not implemented: "
             <<wp<<"\n...break\n"<<std::endl;
    exit(96);
}



std::string JetPileupId::convertWorkingPoint(const JetPileupId::WorkingPoint& wp)
{
    if(wp == none) return "none";
    if(wp == L) return "L";
    if(wp == M) return "M";
    if(wp == T) return "T";
    if(wp == undefinedWP) return "";
    std::cerr<<"Error in JetPileupId::convertWorkingPoint()! Conversion is not implemented\n...break\n"<<std::endl;
    exit(96);
}



int JetPileupId::cutValue(const JetPileupId::WorkingPoint& wp)
{
    if(wp == none) return 0;
    if(wp == L) return 4;
    if(wp == M) return 6;
    if(wp == T) return 7;
    std::cerr<<"Error in JetPileupId::cutValue()! Following conversion is not implemented: "
             <<wp<<"\n...break\n"<<std::endl;
    exit(96);
}





// --------------------- Functions defined in namespace Btag -------------------------

Btag::Algorithm Btag::convertAlgorithm(const std::string& algo)
{
    if(algo == "csv") return csv;
    if(algo == "cmvav2") return cmvav2;
    if(algo == "cmvav2_2016") return cmvav2_2016;    
    if(algo == "csvv2") return csvv2;
    if(algo == "csvv2_2016") return csvv2_2016;    
    if(algo == "csvv2_74X") return csvv2_74X;
    if(algo == "csvv2_50ns") return csvv2_50ns;
    if(algo == "csvv2HIP") return csvv2HIP;
    if(algo == "") return undefinedAlgorithm;
    std::cerr<<"Error in Btag::convertAlgorithm()! Following conversion is not implemented: "
             <<algo<<"\n...break\n"<<std::endl;
    exit(96);
}



std::string Btag::convertAlgorithm(const Btag::Algorithm& algo)
{
    if(algo == csv) return "csv";
    if(algo == cmvav2) return "cmvav2";
    if(algo == cmvav2_2016) return "cmvav2_2016";    
    if(algo == csvv2) return "csvv2";
    if(algo == csvv2_2016) return "csvv2_2016";    
    if(algo == csvv2_74X) return "csvv2_74X";
    if(algo == csvv2_50ns) return "csvv2_50ns";
    if(algo == csvv2HIP) return "csvv2HIP";
    if(algo == undefinedAlgorithm) return "";
    std::cerr<<"Error in Btag::convertAlgorithm()! Conversion is not implemented\n...break\n"<<std::endl;
    exit(96);
}



Btag::WorkingPoint Btag::convertWorkingPoint(const std::string& wp)
{
    if(wp == "L") return L;
    if(wp == "M") return M;
    if(wp == "T") return T;
    if(wp == "") return undefinedWP;
    std::cerr<<"Error in Btag::convertWorkingPoint()! Following conversion is not implemented: "
             <<wp<<"\n...break\n"<<std::endl;
    exit(96);
}



std::string Btag::convertWorkingPoint(const Btag::WorkingPoint& wp)
{
    if(wp == L) return "L";
    if(wp == M) return "M";
    if(wp == T) return "T";
    if(wp == undefinedWP) return "";
    std::cerr<<"Error in Btag::convertWorkingPoint()! Conversion is not implemented\n...break\n"<<std::endl;
    exit(96);
}



Btag::CorrectionMode Btag::convertCorrectionMode(const std::string& mode)
{
    if(mode == "noCorrection") return noCorrection;
    if(mode == "greaterEqualOneTagReweight") return greaterEqualOneTagReweight;
    if(mode == "randomNumberRetag") return randomNumberRetag;
    if(mode == "discriminatorReweight") return discriminatorReweight;
    if(mode == "") return undefinedCorrectionMode;
    std::cerr<<"Error in Btag::convertCorrectionMode()! Following conversion is not implemented: "
             <<mode<<"\n...break\n"<<std::endl;
    exit(96);
}



std::string Btag::convertCorrectionMode(const Btag::CorrectionMode& mode)
{
    if(mode == noCorrection) return "noCorrection";
    if(mode == greaterEqualOneTagReweight) return "greaterEqualOneTagReweight";
    if(mode == randomNumberRetag) return "randomNumberRetag";
    if(mode == discriminatorReweight) return "discriminatorReweight";
    if(mode == undefinedCorrectionMode) return "";
    std::cerr<<"Error in Btag::convertCorrectionMode()! Conversion is not implemented\n...break\n"<<std::endl;
    exit(96);
}





// --------------------- Functions defined in namespace Met -------------------------

Met::Algorithm Met::convertAlgorithm(const std::string& algo)
{
    if(algo == "pf") return pf;
    if(algo == "mva") return mva;
    if(algo == "pfNoHf") return pfNoHf;
    if(algo == "") return undefinedAlgorithm;
    std::cerr<<"Error in Met::convertAlgorithm()! Following conversion is not implemented: "
             <<algo<<"\n...break\n"<<std::endl;
    exit(96);
}



std::string Met::convertAlgorithm(const Met::Algorithm& algo)
{
    if(algo == pf) return "pf";
    if(algo == mva) return "mva";
    if(algo == pfNoHf) return "pfNoHf";
    if(algo == undefinedAlgorithm) return "";
    std::cerr<<"Error in Mva::convertAlgorithm()! Conversion is not implemented\n...break\n"<<std::endl;
    exit(96);
}





// --------------------- Functions defined in namespace Systematic for Type -------------------------



Systematic::Type Systematic::convertType(const TString& type)
{
    // Attention: the order here is important, since the first line where the BeginsWith is true is returned
    if(type.BeginsWith("Nominal")) return nominal;
    if(type.BeginsWith("mH110")) return mH110;
    if(type.BeginsWith("mH115")) return mH115;
    if(type.BeginsWith("mH120")) return mH120;
    if(type.BeginsWith("mH1225")) return mH1225;
    if(type.BeginsWith("mH1275")) return mH1275;
    if(type.BeginsWith("mH130")) return mH130;
    if(type.BeginsWith("mH135")) return mH135;
    if(type.BeginsWith("mH140")) return mH140;
    if(type.BeginsWith("LEPT")) return lept;
    if(type.BeginsWith("ELE")) return ele;
    if(type.BeginsWith("MUON")) return muon;
    if(type.BeginsWith("TRIG_ETA")) return trigEta;
    if(type.BeginsWith("TRIG")) return trig;
    if(type.BeginsWith("PU")) return pu;    
    if(type.BeginsWith("DY")) return dy;
    if(type.BeginsWith("BG")) return bg;
    if(type.BeginsWith("KIN")) return kin;
    if(type.BeginsWith("BTAGDISCR_BSTAT1")) return btagDiscrBstat1;
    if(type.BeginsWith("BTAGDISCR_BSTAT2")) return btagDiscrBstat2;
    if(type.BeginsWith("BTAGDISCR_LSTAT1")) return btagDiscrLstat1;
    if(type.BeginsWith("BTAGDISCR_LSTAT2")) return btagDiscrLstat2;
    if(type.BeginsWith("BTAGDISCR_BPURITY")) return btagDiscrBpurity;
    if(type.BeginsWith("BTAGDISCR_LPURITY")) return btagDiscrLpurity;
    if(type.BeginsWith("BTAGDISCR_CERR1")) return btagDiscrCerr1;
    if(type.BeginsWith("BTAGDISCR_CERR2")) return btagDiscrCerr2;
    if(type.BeginsWith("BTAG_LJET_PT")) return btagLjetPt;
    if(type.BeginsWith("BTAG_LJET_ETA")) return btagLjetEta;
    if(type.BeginsWith("BTAG_LJET")) return btagLjet;
    if(type.BeginsWith("BTAG_BEFF")) return btagBeff;
    if(type.BeginsWith("BTAG_CEFF")) return btagCeff;
    if(type.BeginsWith("BTAG_LEFF")) return btagLeff;
    if(type.BeginsWith("BTAG_PT")) return btagPt;
    if(type.BeginsWith("BTAG_ETA")) return btagEta;
    if(type.BeginsWith("BTAG")) return btag;
    if(type.BeginsWith("JER")) return jer;
    if(type.BeginsWith("JESAbsoluteStat")) return jesAbsoluteStat;
    if(type.BeginsWith("JESAbsoluteScale")) return jesAbsoluteScale;
    if(type.BeginsWith("JESAbsoluteFlavMap")) return jesAbsoluteFlavMap;
    if(type.BeginsWith("JESAbsoluteMPFBias")) return jesAbsoluteMPFBias;
    if(type.BeginsWith("JESFragmentation")) return jesFragmentation;
    if(type.BeginsWith("JESSinglePionECAL")) return jesSinglePionECAL;
    if(type.BeginsWith("JESSinglePionHCAL")) return jesSinglePionHCAL;
    if(type.BeginsWith("JESFlavorQCD")) return jesFlavorQCD;
    if(type.BeginsWith("JESTimePtEta")) return jesTimePtEta;
//    if(type.BeginsWith("JESTimePt")) return jesTimePt;
    if(type.BeginsWith("JESRelativeJEREC1")) return jesRelativeJEREC1;
    if(type.BeginsWith("JESRelativeJEREC2")) return jesRelativeJEREC2;
    if(type.BeginsWith("JESRelativeJERHF")) return jesRelativeJERHF;
    if(type.BeginsWith("JESRelativePtBB")) return jesRelativePtBB;
    if(type.BeginsWith("JESRelativePtEC1")) return jesRelativePtEC1;
    if(type.BeginsWith("JESRelativePtEC2")) return jesRelativePtEC2;
    if(type.BeginsWith("JESRelativePtHF")) return jesRelativePtHF;
    if(type.BeginsWith("JESRelativeBal")) return jesRelativeBal;    
    if(type.BeginsWith("JESRelativeFSR")) return jesRelativeFSR;
    if(type.BeginsWith("JESRelativeStatFSR")) return jesRelativeStatFSR;
    if(type.BeginsWith("JESRelativeStatEC")) return jesRelativeStatEC;
    if(type.BeginsWith("JESRelativeStatHF")) return jesRelativeStatHF;
    if(type.BeginsWith("JESPileUpDataMC")) return jesPileUpDataMC;
    if(type.BeginsWith("JESPileUpPtRef")) return jesPileUpPtRef;
    if(type.BeginsWith("JESPileUpPtBB")) return jesPileUpPtBB;
    if(type.BeginsWith("JESPileUpPtEC1")) return jesPileUpPtEC1;
    if(type.BeginsWith("JESPileUpPtEC2")) return jesPileUpPtEC2;
    if(type.BeginsWith("JESPileUpPtHF")) return jesPileUpPtHF;
            //"PileUpBias,       //23-----
    if(type.BeginsWith("JESPileUpMuZero")) return jesPileUpMuZero;
    if(type.BeginsWith("JESPileUpEnvelope")) return jesPileUpEnvelope;
    if(type.BeginsWith("JESSubTotalPileUp")) return jesSubTotalPileUp;
    if(type.BeginsWith("JESSubTotalRelative")) return jesSubTotalRelative;
    if(type.BeginsWith("JESSubTotalPt")) return jesSubTotalPt;
    if(type.BeginsWith("JESSubTotalScale")) return jesSubTotalScale;
    if(type.BeginsWith("JESSubTotalMC")) return jesSubTotalMC;
    if(type.BeginsWith("JESSubTotalAbsolute")) return jesSubTotalAbsolute;
    if(type.BeginsWith("JESTotalNoFlavor")) return jesTotalNoFlavor;
            //"TotalNoTime"       // ignoring for the moment these subtotals
            //"TotalNoFlavorNoTime"
            //"Time A-D
    if(type.BeginsWith("JESFlavorZJet")) return jesFlavorZJet;
    if(type.BeginsWith("JESFlavorPhotonJet")) return jesFlavorPhotonJet;
    if(type.BeginsWith("JESFlavorPureGluon")) return jesFlavorPureGluon;
    if(type.BeginsWith("JESFlavorPureQuark")) return jesFlavorPureQuark;
    if(type.BeginsWith("JESFlavorPureCharm")) return jesFlavorPureCharm;
    if(type.BeginsWith("JESFlavorPureBottom")) return jesFlavorPureBottom;
    if(type.BeginsWith("JESCorrelationGroupMPFInSitu")) return jesCorrelationGroupMPFInSitu;
    if(type.BeginsWith("JESCorrelationGroupIntercalibration")) return jesCorrelationGroupIntercalibration;
    if(type.BeginsWith("JESCorrelationGroupbJES")) return jesCorrelationGroupbJES;
    if(type.BeginsWith("JESCorrelationGroupFlavor")) return jesCorrelationGroupFlavor;
    if(type.BeginsWith("JESCorrelationGroupUncorrelated")) return jesCorrelationGroupUncorrelated;
    if(type.BeginsWith("JES")) return jes;
    if(type.BeginsWith("FRAC_TTHF")) return frac_tthf;
    if(type.BeginsWith("FRAC_TTOTHER")) return frac_ttother;
    if(type.BeginsWith("LUMI")) return lumi;
    if(type.BeginsWith("XSEC_TTBB")) return xsec_ttbb;
    if(type.BeginsWith("XSEC_TTB")) return xsec_ttb;
    if(type.BeginsWith("XSEC_TT2B")) return xsec_tt2b;
    if(type.BeginsWith("XSEC_TTCC")) return xsec_ttcc;
    if(type.BeginsWith("XSEC_TTOTHER")) return xsec_ttother;
    if(type.BeginsWith("XSEC_TTZ")) return xsec_ttZ;
    if(type.BeginsWith("XSEC_TTW")) return xsec_ttW;
    if(type.BeginsWith("XSEC_TTG")) return xsec_ttG;
    if(type.BeginsWith("XSEC_TTH")) return xsec_ttH;
    if(type.BeginsWith("XSEC_TTDM")) return xsec_ttDM;
    if(type.BeginsWith("XSEC_TT")) return xsec_tt;
    if(type.BeginsWith("XSEC_T")) return xsec_t;
    if(type.BeginsWith("XSEC_VV")) return xsec_vv;
    if(type.BeginsWith("XSEC_V")) return xsec_v;
    if(type.BeginsWith("XSEC_W")) return xsec_w;
    if(type.BeginsWith("XSEC_Z")) return xsec_z;
    if(type.BeginsWith("TOP_PT_THEORY")) return topPtTheory; 
    if(type.BeginsWith("TOP_PT_FITPARAM0")) return topPtFitparam0;
    if(type.BeginsWith("TOP_PT_FITPARAM1")) return topPtFitparam1;           
    if(type.BeginsWith("TOP_PT")) return topPt;
    if(type.BeginsWith("MASS")) return mass;
    if(type.BeginsWith("MATCH")) return match;
    if(type.BeginsWith("GLUONMOVETUNE")) return gluonmovetune;
    if(type.BeginsWith("ERDONRETUNE")) return erdonretune;
    if(type.BeginsWith("ERDON")) return erdon;
    if(type.BeginsWith("MESCALE_TTBB")) return meScale_ttbb;
    if(type.BeginsWith("MESCALE_TTB")) return meScale_ttb;
    if(type.BeginsWith("MESCALE_TT2B")) return meScale_tt2b;
    if(type.BeginsWith("MESCALE_TTCC")) return meScale_ttcc;
    if(type.BeginsWith("MESCALE_TTOTHER")) return meScale_ttother;
    if(type.BeginsWith("MESCALE")) return meScale;
    if(type.BeginsWith("MEFACSCALE_TTBB")) return meFacScale_ttbb;
    if(type.BeginsWith("MEFACSCALE_TTB")) return meFacScale_ttb;
    if(type.BeginsWith("MEFACSCALE_TT2B")) return meFacScale_tt2b;
    if(type.BeginsWith("MEFACSCALE_TTCC")) return meFacScale_ttcc;
    if(type.BeginsWith("MEFACSCALE_TTOTHER")) return meFacScale_ttother;
    if(type.BeginsWith("MEFACSCALE_TTV")) return meFacScale_ttv;
    if(type.BeginsWith("MEFACSCALE_TTZ")) return meFacScale_ttz;
    if(type.BeginsWith("MEFACSCALE_TTW")) return meFacScale_ttw;
    if(type.BeginsWith("MEFACSCALE_TTG")) return meFacScale_ttg;
    if(type.BeginsWith("MEFACSCALE_TTDM")) return meFacScale_ttdm;
    if(type.BeginsWith("MEFACSCALE_HTOTT_RES")) return meFacScale_htott_res;
    if(type.BeginsWith("MEFACSCALE_HTOTT_INT")) return meFacScale_htott_int;       
    if(type.BeginsWith("MEFACSCALE_TT")) return meFacScale_tt;
    if(type.BeginsWith("MEFACSCALE_Z")) return meFacScale_z;
    if(type.BeginsWith("MEFACSCALE_W")) return meFacScale_w;
    if(type.BeginsWith("MEFACSCALE_ST")) return meFacScale_st;
    if(type.BeginsWith("MEFACSCALE_VV")) return meFacScale_vv;
    if(type.BeginsWith("MEFACSCALE")) return meFacScale;  
    if(type.BeginsWith("MERENSCALE_TTBB")) return meRenScale_ttbb;
    if(type.BeginsWith("MERENSCALE_TTB")) return meRenScale_ttb;
    if(type.BeginsWith("MERENSCALE_TT2B")) return meRenScale_tt2b;
    if(type.BeginsWith("MERENSCALE_TTCC")) return meRenScale_ttcc;
    if(type.BeginsWith("MERENSCALE_TTOTHER")) return meRenScale_ttother;
    if(type.BeginsWith("MERENSCALE_TTV")) return meRenScale_ttv;
    if(type.BeginsWith("MERENSCALE_TTZ")) return meRenScale_ttz;
    if(type.BeginsWith("MERENSCALE_TTW")) return meRenScale_ttw;
    if(type.BeginsWith("MERENSCALE_TTG")) return meRenScale_ttg;
    if(type.BeginsWith("MERENSCALE_TTDM")) return meRenScale_ttdm;
    if(type.BeginsWith("MERENSCALE_HTOTT_RES")) return meRenScale_htott_res; 
    if(type.BeginsWith("MERENSCALE_HTOTT_INT")) return meRenScale_htott_int;
    if(type.BeginsWith("MERENSCALE_TT")) return meRenScale_tt;
    if(type.BeginsWith("MERENSCALE_Z")) return meRenScale_z;
    if(type.BeginsWith("MERENSCALE_W")) return meRenScale_w;
    if(type.BeginsWith("MERENSCALE_ST")) return meRenScale_st;
    if(type.BeginsWith("MERENSCALE_VV")) return meRenScale_vv;
    if(type.BeginsWith("MERENSCALE")) return meRenScale;      
    if(type.BeginsWith("PSSCALE_TTBB")) return psScale_ttbb;
    if(type.BeginsWith("PSSCALE_TTB")) return psScale_ttb;
    if(type.BeginsWith("PSSCALE_TT2B")) return psScale_tt2b;
    if(type.BeginsWith("PSSCALE_TTCC")) return psScale_ttcc;
    if(type.BeginsWith("PSSCALE_TTOTHER")) return psScale_ttother;
    if(type.BeginsWith("PSSCALE")) return psScale;
    if(type.BeginsWith("PSISRSCALE")) return psISRScale;
    if(type.BeginsWith("PSFSRSCALE")) return psFSRScale;
    if(type.BeginsWith("SCALE_TTBB")) return scale_ttbb;
    if(type.BeginsWith("SCALE_TTB")) return scale_ttb;
    if(type.BeginsWith("SCALE_TT2B")) return scale_tt2b;
    if(type.BeginsWith("SCALE_TTCC")) return scale_ttcc;
    if(type.BeginsWith("SCALE_TTOTHER")) return scale_ttother;
    if(type.BeginsWith("SCALE")) return scale;
    if(type.BeginsWith("BFRAG_CENTRAL")) return bFrag_central;  
    if(type.BeginsWith("BFRAG_PETERSON")) return bFrag_Peterson;  
    if(type.BeginsWith("BFRAG")) return bFrag;  
    if(type.BeginsWith("BSEMILEP")) return bSemilep;  
    if(type.BeginsWith("UETUNE")) return ueTune;
    if(type.BeginsWith("POWHEGV2HERWIG")) return powhegv2Herwig;
    if(type.BeginsWith("POWHEGHERWIG")) return powhegHerwig;
    if(type.BeginsWith("POWHEGV2")) return powhegv2;
    if(type.BeginsWith("POWHEG")) return powheg;
    if(type.BeginsWith("AMCATNLOFXFX")) return amcatnlofxfx;
    if(type.BeginsWith("MCATNLO")) return mcatnlo;
    if(type.BeginsWith("MADGRAPHMLM")) return madgraphmlm;
    if(type.BeginsWith("PERUGIA11NoCR")) return perugia11NoCR;
    if(type.BeginsWith("PERUGIA11")) return perugia11;
    if(type.BeginsWith("PDF_ALPHAS")) return alphasPdf;
    if(type.BeginsWith("NORMPDFGG")) return normPdfGg;
    if(type.BeginsWith("NORMPDFGQ")) return normPdfGq;
    if(type.BeginsWith("NORMPDFQQ")) return normPdfQq;
    if(type.BeginsWith("NORMPDFTTH")) return normPdfTth;
    if(type.BeginsWith("PDF")) return pdf;
    if(type.BeginsWith("closure")) return closure;
    if(type.BeginsWith("allAvailable")) return allAvailable;
    if(type.BeginsWith("all")) return all;
    std::cout<<"Warning in Systematic::convertType()! Following conversion is not implemented: "
             <<type<<std::endl<<std::endl;
    return undefinedType;
}



TString Systematic::convertType(const Type& type)
{
    if(type == nominal) return "Nominal";
    if(type == mH110) return "mH110";
    if(type == mH115) return "mH115";
    if(type == mH120) return "mH120";
    if(type == mH1225) return "mH1225";
    if(type == mH1275) return "mH1275";
    if(type == mH130) return "mH130";
    if(type == mH135) return "mH135";
    if(type == mH140) return "mH140";
    if(type == lept) return "LEPT";
    if(type == ele) return "ELE";
    if(type == muon) return "MUON";
    if(type == trigEta) return "TRIG_ETA";
    if(type == trig) return "TRIG";
    if(type == pu) return "PU";
    if(type == dy) return "DY";
    if(type == bg) return "BG";
    if(type == kin) return "KIN";
    if(type == btagDiscrBstat1) return "BTAGDISCR_BSTAT1";
    if(type == btagDiscrBstat2) return "BTAGDISCR_BSTAT2";
    if(type == btagDiscrLstat1) return "BTAGDISCR_LSTAT1";
    if(type == btagDiscrLstat2) return "BTAGDISCR_LSTAT2";
    if(type == btagDiscrBpurity) return "BTAGDISCR_BPURITY";
    if(type == btagDiscrLpurity) return "BTAGDISCR_LPURITY";
    if(type == btagDiscrCerr1) return "BTAGDISCR_CERR1";
    if(type == btagDiscrCerr2) return "BTAGDISCR_CERR2";
    if(type == btagLjetPt) return "BTAG_LJET_PT";
    if(type == btagLjetEta) return "BTAG_LJET_ETA";
    if(type == btagLjet) return "BTAG_LJET";
    if(type == btagBeff) return "BTAG_BEFF";
    if(type == btagCeff) return "BTAG_CEFF";
    if(type == btagLeff) return "BTAG_LEFF";
    if(type == btagPt) return "BTAG_PT";
    if(type == btagEta) return "BTAG_ETA";
    if(type == btag) return "BTAG";
    if(type == jer) return "JER";
    if(type == jes) return "JES";
    if(type == jesAbsoluteStat) return "JESAbsoluteStat";
    if(type == jesAbsoluteScale) return "JESAbsoluteScale";
    if(type == jesAbsoluteFlavMap) return "JESAbsoluteFlavMap";
    if(type == jesAbsoluteMPFBias) return "JESAbsoluteMPFBias";
    if(type == jesFragmentation) return "JESFragmentation";
    if(type == jesSinglePionECAL) return "JESSinglePionECAL";
    if(type == jesSinglePionHCAL) return "JESSinglePionHCAL";
    if(type == jesFlavorQCD) return "JESFlavorQCD";
    if(type == jesTimePtEta) return "JESTimePtEta";
    if(type == jesRelativeJEREC1) return "JESRelativeJEREC1";
    if(type == jesRelativeJEREC2) return "JESRelativeJEREC2";
    if(type == jesRelativeJERHF) return "JESRelativeJERHF";
    if(type == jesRelativePtBB) return "JESRelativePtBB";
    if(type == jesRelativePtEC1) return "JESRelativePtEC1";
    if(type == jesRelativePtEC2) return "JESRelativePtEC2";
    if(type == jesRelativePtHF) return "JESRelativePtHF";
    if(type == jesRelativeBal) return "JESRelativeBal";    
    if(type == jesRelativeFSR) return "JESRelativeFSR";
    if(type == jesRelativeStatFSR) return "JESRelativeStatFSR";
    if(type == jesRelativeStatEC) return "JESRelativeStatEC";
    if(type == jesRelativeStatHF) return "JESRelativeStatHF";
    if(type == jesPileUpDataMC) return "JESPileUpDataMC";
    if(type == jesPileUpPtRef) return "JESPileUpPtRef";
    if(type == jesPileUpPtBB) return "JESPileUpPtBB";
    if(type == jesPileUpPtEC1) return "JESPileUpPtEC1";
    if(type == jesPileUpPtEC2) return "JESPileUpPtEC2";
    if(type == jesPileUpPtHF) return "JESPileUpPtHF";
            //"PileUpBias,       //23-----
    if(type == jesPileUpMuZero) return "JESPileUpMuZero";
    if(type == jesPileUpEnvelope) return "JESPileUpEnvelope";
    if(type == jesSubTotalPileUp) return "JESSubTotalPileUp";
    if(type == jesSubTotalRelative) return "JESSubTotalRelative";
    if(type == jesSubTotalPt) return "JESSubTotalPt";
    if(type == jesSubTotalScale) return "JESSubTotalScale";
    if(type == jesSubTotalMC) return "JESSubTotalMC";
    if(type == jesSubTotalAbsolute) return "JESSubTotalAbsolute";
    if(type == jesTotalNoFlavor) return "JESTotalNoFlavor";
            //"TotalNoTime"       // ignoring for the moment these subtotals
            //"TotalNoFlavorNoTime"
            //"Time A-D
    if(type == jesFlavorZJet) return "JESFlavorZJet";
    if(type == jesFlavorPhotonJet) return "JESFlavorPhotonJet";
    if(type == jesFlavorPureGluon) return "JESFlavorPureGluon";
    if(type == jesFlavorPureQuark) return "JESFlavorPureQuark";
    if(type == jesFlavorPureCharm) return "JESFlavorPureCharm";
    if(type == jesFlavorPureBottom) return "JESFlavorPureBottom";
    if(type == jesCorrelationGroupMPFInSitu) return "JESCorrelationGroupMPFInSitu";
    if(type == jesCorrelationGroupIntercalibration) return "JESCorrelationGroupIntercalibration";
    if(type == jesCorrelationGroupbJES) return "JESCorrelationGroupbJES";
    if(type == jesCorrelationGroupFlavor) return "JESCorrelationGroupFlavor";
    if(type == jesCorrelationGroupUncorrelated) return "JESCorrelationGroupUncorrelated";
    if(type == frac_tthf) return "FRAC_TTHF";
    if(type == frac_ttother) return "FRAC_TTOTHER";
    if(type == lumi) return "LUMI";
    if(type == xsec_ttbb) return "XSEC_TTBB";
    if(type == xsec_ttb) return "XSEC_TTB";
    if(type == xsec_tt2b) return "XSEC_TT2B";
    if(type == xsec_ttcc) return "XSEC_TTCC";
    if(type == xsec_ttother) return "XSEC_TTOTHER";
    if(type == xsec_ttZ) return "XSEC_TTZ";
    if(type == xsec_ttW) return "XSEC_TTW";
    if(type == xsec_ttG) return "XSEC_TTG";
    if(type == xsec_ttH) return "XSEC_TTH";
    if(type == xsec_ttDM) return "XSEC_TTDM";
    if(type == xsec_tt) return "XSEC_TT";
    if(type == xsec_t) return "XSEC_T";
    if(type == xsec_vv) return "XSEC_VV";
    if(type == xsec_v) return "XSEC_V";
    if(type == xsec_w) return "XSEC_W";
    if(type == xsec_z) return "XSEC_Z";
    if(type == topPtTheory) return "TOP_PT_THEORY";
    if(type == topPtFitparam0) return "TOP_PT_FITPARAM0"; 
    if(type == topPtFitparam1) return "TOP_PT_FITPARAM1";             
    if(type == topPt) return "TOP_PT";
    if(type == mass) return "MASS";
    if(type == match) return "MATCH";
    if(type == gluonmovetune) return "GLUONMOVETUNE";
    if(type == erdonretune) return "ERDONRETUNE";
    if(type == erdon) return "ERDON";
    if(type == meScale_ttbb) return "MESCALE_TTBB";
    if(type == meScale_ttb) return "MESCALE_TTB";
    if(type == meScale_tt2b) return "MESCALE_TT2B";
    if(type == meScale_ttcc) return "MESCALE_TTCC";
    if(type == meScale_ttother) return "MESCALE_TTOTHER";
    if(type == meScale) return "MESCALE";
    if(type == meFacScale_ttbb) return "MEFACSCALE_TTBB";
    if(type == meFacScale_ttb) return "MEFACSCALE_TTB";
    if(type == meFacScale_tt2b) return "MEFACSCALE_TT2B";
    if(type == meFacScale_ttcc) return "MEFACSCALE_TTCC";
    if(type == meFacScale_ttother) return "MEFACSCALE_TTOTHER";
    if(type == meFacScale_tt) return "MEFACSCALE_TT";
    if(type == meFacScale_z) return "MEFACSCALE_Z";
    if(type == meFacScale_w) return "MEFACSCALE_W";
    if(type == meFacScale_st) return "MEFACSCALE_ST";
    if(type == meFacScale_vv) return "MEFACSCALE_VV";
    if(type == meFacScale_ttv) return "MEFACSCALE_TTV";
    if(type == meFacScale_ttz) return "MEFACSCALE_TTZ";
    if(type == meFacScale_ttw) return "MEFACSCALE_TTW";
    if(type == meFacScale_ttg) return "MEFACSCALE_TTG";
    if(type == meFacScale_ttdm) return "MEFACSCALE_TTDM";
    if(type == meFacScale_htott_res) return "MEFACSCALE_HTOTT_RES";
    if(type == meFacScale_htott_int) return "MEFACSCALE_HTOTT_INT";  
    if(type == meFacScale) return "MEFACSCALE"; 
    if(type == meRenScale_ttbb) return "MERENSCALE_TTBB";
    if(type == meRenScale_ttb) return "MERENSCALE_TTB";
    if(type == meRenScale_tt2b) return "MERENSCALE_TT2B";
    if(type == meRenScale_ttcc) return "MERENSCALE_TTCC";
    if(type == meRenScale_ttother) return "MERENSCALE_TTOTHER";
    if(type == meRenScale_tt) return "MERENSCALE_TT";
    if(type == meRenScale_z) return "MERENSCALE_Z";
    if(type == meRenScale_w) return "MERENSCALE_W";
    if(type == meRenScale_st) return "MERENSCALE_ST";
    if(type == meRenScale_vv) return "MERENSCALE_VV";
    if(type == meRenScale_ttv) return "MERENSCALE_TTV";
    if(type == meRenScale_ttz) return "MERENSCALE_TTZ";
    if(type == meRenScale_ttw) return "MERENSCALE_TTW";
    if(type == meRenScale_ttg) return "MERENSCALE_TTG";
    if(type == meRenScale_ttdm) return "MERENSCALE_TTDM";
    if(type == meRenScale_htott_res) return "MERENSCALE_HTOTT_RES";  
    if(type == meRenScale_htott_int) return "MERENSCALE_HTOTT_INT"; 
    if(type == meRenScale) return "MERENSCALE";       
    if(type == psScale_ttbb) return "PSSCALE_TTBB";
    if(type == psScale_ttb) return "PSSCALE_TTB";
    if(type == psScale_tt2b) return "PSSCALE_TT2B";
    if(type == psScale_ttcc) return "PSSCALE_TTCC";
    if(type == psScale_ttother) return "PSSCALE_TTOTHER";
    if(type == psScale) return "PSSCALE";
    if(type == psISRScale) return "PSISRSCALE";
    if(type == psFSRScale) return "PSFSRSCALE";
    if(type == scale_ttbb) return "SCALE_TTBB";
    if(type == scale_ttb) return "SCALE_TTB";
    if(type == scale_tt2b) return "SCALE_TT2B";
    if(type == scale_ttcc) return "SCALE_TTCC";
    if(type == scale_ttother) return "SCALE_TTOTHER";
    if(type == scale) return "SCALE";
    if(type == bFrag) return "BFRAG";
    if(type == bFrag_central) return "BFRAG_CENTRAL";
    if(type == bFrag_Peterson) return "BFRAG_PETERSON";
    if(type == bSemilep) return "BSEMILEP";
    if(type == ueTune) return "UETUNE";
    if(type == powhegv2Herwig) return "POWHEGV2HERWIG";
    if(type == powhegHerwig) return "POWHEGHERWIG";
    if(type == powhegv2) return "POWHEGV2";
    if(type == powheg) return "POWHEG";
    if(type == amcatnlofxfx) return "AMCATNLOFXFX";
    if(type == mcatnlo) return "MCATNLO";
    if(type == madgraphmlm) return "MADGRAPHMLM";
    if(type == perugia11NoCR) return "PERUGIA11NoCR";
    if(type == perugia11) return "PERUGIA11";
    if(type == alphasPdf) return "PDF_ALPHAS";
    if(type == normPdfGg) return "NORMPDFGG";
    if(type == normPdfGq) return "NORMPDFGQ";
    if(type == normPdfQq) return "NORMPDFQQ";
    if(type == normPdfTth) return "NORMPDFTTH";
    if(type == pdf) return "PDF";
    if(type == closure) return "closure";
    if(type == allAvailable) return "allAvailable";
    if(type == all) return "all";
    if(type == undefinedType) return "";
    std::cerr<<"Error in Systematic::convertType()! Conversion is not implemented\n...break\n"<<std::endl;
    exit(99);
}



std::vector<Systematic::Type> Systematic::convertType(const std::vector<TString>& types)
{
    std::vector<Type> v_type;
    for(const auto& type : types) v_type.push_back(convertType(type));
    return v_type;
}



std::vector<Systematic::Type> Systematic::convertType(const std::vector<std::string>& types)
{
    std::vector<Type> v_type;
    for(const auto& type : types) v_type.push_back(convertType(type));
    return v_type;
}



std::vector<TString> Systematic::convertType(const std::vector<Type>& types)
{
    std::vector<TString> v_type;
    for(const auto& type : types) v_type.push_back(convertType(type));
    return v_type;
}










// --------------------- Functions defined in namespace Systematic for Variation -------------------------





Systematic::Variation Systematic::convertVariation(const TString& variation)
{
    if(variation.EndsWith("_UP")) return up;
    if(variation.EndsWith("_DOWN")) return down;
    if(variation.EndsWith("_CENTRAL")) return central;
    //std::cout<<"Warning in Systematic::convertVariation()! Following conversion is not implemented: "
    //         <<variation<<std::endl<<std::endl;
    return undefinedVariation;
}



TString Systematic::convertVariation(const Variation& variation)
{
    if(variation == up) return "_UP";
    if(variation == down) return "_DOWN";
    if(variation == central) return "_CENTRAL";
    if(variation == undefinedVariation) return "";
    std::cerr<<"Error in Systematic::convertVariation()! Conversion is not implemented\n...break\n"<<std::endl;
    exit(99);
}



std::vector<Systematic::Variation> Systematic::convertVariation(const std::vector<TString>& variations)
{
    std::vector<Variation> v_variation;
    for(const auto& variation : variations) v_variation.push_back(convertVariation(variation));
    return v_variation;
}



std::vector<Systematic::Variation> Systematic::convertVariation(const std::vector<std::string>& variations)
{
    std::vector<Variation> v_variation;
    for(const auto& variation : variations) v_variation.push_back(convertVariation(variation));
    return v_variation;
}



std::vector<TString> Systematic::convertVariation(const std::vector<Variation>& variations)
{
    std::vector<TString> v_variation;
    for(const auto& variation : variations) v_variation.push_back(convertVariation(variation));
    return v_variation;
}










// --------------------- Further functions defined in namespace Systematic -------------------------



void Systematic::isValid(const Type& type, const Variation& variation, const int variationNumber)
{
    // Check validity of variationNumber
    if(variationNumber >= 0){
        if(std::find(centralTypes.begin(), centralTypes.end(), type) == centralTypes.end()){
            std::cerr<<"ERROR in Systematic::isValid()! Given type does not allow variation numbers (type, variationNumber): "
                     <<convertType(type)<<" , "<<variationNumber<<"\n...break\n"<<std::endl;
            exit(7);
        }
    }
    
    // Check validity of variation
    if(variation == undefinedVariation) return;
    else if(variation==up || variation==down){
        if(std::find(upDownTypes.begin(), upDownTypes.end(), type) == upDownTypes.end()){
            std::cerr<<"ERROR in Systematic::isValid()! Given type does not allow variation (type, variation): "
                     <<convertType(type)<<" , "<<convertVariation(variation)<<"\n...break\n"<<std::endl;
            exit(7);
        }
    }
    else if(variation == central){
        if(std::find(centralTypes.begin(), centralTypes.end(), type) == centralTypes.end()){
            std::cerr<<"ERROR in Systematic::isValid()! Given type does not allow variation (type, variation): "
                     <<convertType(type)<<" , "<<convertVariation(variation)<<"\n...break\n"<<std::endl;
            exit(7);
        }
    }
    else{
        std::cerr<<"ERROR in Systematic::isValid()! Variation is not defined for validity check: "
                 <<convertVariation(variation)<<"\n...break\n"<<std::endl;
        exit(7);
    }
}




std::vector<Systematic::Systematic> Systematic::allowedSystematicsAnalysis(const std::vector<Type>& allowedTypes)
{
    std::vector<Systematic> result;
    
    for(const Type& type : allowedTypes){
        // Exclude non-real types
        if(type==all || type==allAvailable) continue;
        
        if(std::find(centralTypes.begin(), centralTypes.end(), type) != centralTypes.end()){
            // Central types need specific treatment using variation numbers, e.g. PDF variations
            // They require detailed specifications at the place where they are used
            // FIXME: This implementation of 26 hardcoded PDF variations should be made more generic
            if(type == pdf){
                result.push_back(Systematic(type, central, 0));
                for(int id = 1; id <= 26; ++id){
                    result.push_back(Systematic(type, up, id));
                    result.push_back(Systematic(type, down, id));
                }
            }
            else
                result.push_back(Systematic(type, undefinedVariation));
        }
        else if(std::find(upDownTypes.begin(), upDownTypes.end(), type) != upDownTypes.end()){
            // Up/down types need the two variations
            result.push_back(Systematic(type, up));
            result.push_back(Systematic(type, down));
        }
        else{
            // All others have no variations
            result.push_back(Systematic(type, undefinedVariation));
        }
    }
    
    return result;
}



std::vector<Systematic::Systematic> Systematic::setSystematics(const std::vector<std::string>& systematicNames)
{
    std::vector<Systematic> result;
    
    for(const auto& name : systematicNames) result.push_back(Systematic(name));
    
    return result;
}


Systematic::Systematic Systematic::nominalSystematic()
{
    return Systematic(nominal, undefinedVariation);
}



Systematic::Systematic Systematic::undefinedSystematic()
{
    return Systematic(undefinedType, undefinedVariation);
}




// --------------------- Methods of class Systematic in namespace Systematic -------------------------



Systematic::Systematic::Systematic():
type_(undefinedType),
variation_(undefinedVariation),
variationNumber_(-1)
{}



Systematic::Systematic::Systematic(const Type& type, const Variation& variation, const int variationNumber):
type_(type),
variation_(variation),
variationNumber_(variationNumber)
{}



Systematic::Systematic::Systematic(const TString& systematicName):
type_(undefinedType),
variation_(undefinedVariation),
variationNumber_(-1)
{

    TString fragment(systematicName);
    type_ = convertType(systematicName);
    fragment.ReplaceAll(convertType(type_), "");
    variation_ = convertVariation(fragment);
    fragment.ReplaceAll(convertVariation(variation_), "");
    if(fragment != ""){
        fragment.ReplaceAll("_", "");
        int variationNumber = -1;
        std::stringstream stream(fragment.Data());
        if(!(stream>>variationNumber)){
            std::cerr<<"ERROR in constructor of Systematic! Could not fragment systematic name (name --- type, variation, variationNumber): "
                     <<systematicName<<" --- "<<convertType(type_)<<" , "<<convertVariation(variation_)<<" , "<<variationNumber_<<"\n...break\n"<<std::endl;
            exit(8);
        }
        variationNumber_ = variationNumber;
        if(variationNumber_ < 0){
            std::cerr<<"ERROR in constructor of Systematic! Variation numbers must be >=0, but is (systematicName, extracted variationNumber): "
                     <<systematicName<<" , "<<variationNumber_<<"\n...break\n"<<std::endl;
            exit(8);
        }
    }
    isValid(type_, variation_, variationNumber_);
}



TString Systematic::Systematic::name()const
{
    TString result = convertType(type_);
    if(variationNumber_ >= 0){
        std::stringstream stream;
        stream<<"_"<<variationNumber_;
        result.Append(stream.str());
    }
    result.Append(convertVariation(variation_));
    return result;
}










// --------------------- Functions defined in namespace Channel -------------------------



Channel::Channel Channel::convert(const TString& channel)
{
    if(channel == "ee") return ee;
    if(channel == "emu") return emu;
    if(channel == "mumu") return mumu;
    if(channel == "combined") return combined;
    if(channel == "tautau") return tautau;
    if(channel == "se") return se;
    if(channel == "smu") return smu;
    if(channel == "") return undefined;
    std::cerr<<"Error in Channel::convert()! Following conversion is not implemented: "<<channel<<"\n...break\n"<<std::endl;
    exit(98);
}



TString Channel::convert(const Channel& channel)
{
    if(channel == ee) return "ee";
    if(channel == emu) return "emu";
    if(channel == mumu) return "mumu";
    if(channel == combined) return "combined";
    if(channel == tautau) return "tautau";
    if(channel == se) return "se";
    if(channel == smu) return "smu";
    if(channel == undefined) return "";
    std::cerr<<"Error in Channel::convert()! Conversion is not implemented\n...break\n"<<std::endl;
    exit(98);
}



TString Channel::label(const Channel& channel)
{
    if(channel == ee) return "ee";
    if(channel == emu) return "e#mu";
    if(channel == mumu) return "#mu#mu";
    if(channel == combined) return "Dilepton Combined";
    if(channel == tautau) return "#tau#tau";
    if(channel == se) return "From single e";
    if(channel == smu) return "From single #mu";
    if(channel == undefined) return "";
    std::cerr<<"Error! Channel label is not implemented,\n...break\n"<<std::endl;
    exit(98);
}



std::vector<Channel::Channel> Channel::convert(const std::vector<TString>& channels)
{
    std::vector<Channel> v_channel;
    for(const auto& channel : channels) v_channel.push_back(convert(channel));
    return v_channel;
}



std::vector<Channel::Channel> Channel::convert(const std::vector<std::string>& channels)
{
    std::vector<Channel> v_channel;
    for(const auto& channel : channels) v_channel.push_back(convert(channel));
    return v_channel;
}



std::vector<TString> Channel::convert(const std::vector<Channel>& channels)
{
    std::vector<TString> v_channel;
    for(const auto& channel : channels) v_channel.push_back(convert(channel));
    return v_channel;
}









// --------------------- Functions defined in namespace common -------------------------



TString common::assignFolder(const char* baseDir, const Channel::Channel& channel, const Systematic::Systematic& systematic, const char* subDir)
{
    TString path("");
    
    // Create all subdirectories contained in baseDir
    TObjArray* a_currentDir = TString(baseDir).Tokenize("/");
    for(Int_t iCurrentDir = 0; iCurrentDir < a_currentDir->GetEntriesFast(); ++iCurrentDir){
        const TString& currentDir = a_currentDir->At(iCurrentDir)->GetName();
        path.Append(currentDir);
        path.Append("/");
        gSystem->MakeDirectory(path);
    }
    
    // Create subdirectories for systematic and channel
    path.Append(systematic.name());
    path.Append("/");
    gSystem->MakeDirectory(path);
    path.Append(Channel::convert(channel));
    path.Append("/");
    gSystem->MakeDirectory(path);
    if(TString(subDir) != ""){
        path.Append(subDir);
        path.Append("/");
    }
    gSystem->MakeDirectory(path);
    
    return path;
}



TString common::accessFolder(const char* baseDir, const Channel::Channel& channel,
                             const Systematic::Systematic& systematic, const bool allowNonexisting)
{
    // Build directory path
    TString path(baseDir);
    path.Append("/");
    path.Append(systematic.name());
    path.Append("/");
    path.Append(Channel::convert(channel));
    path.Append("/");
    
    // Check if directory really exists
    if(!gSystem->OpenDirectory(path)){
        if(allowNonexisting){
            // It is allowed to request a folder which does not exist, so return empty string silently
            return "";
        }
        else{
            std::cerr<<"ERROR! Request to access directory is not possible, because it does not exist. Directory name: "
                     <<path<<"\n...break\n"<<std::endl;
            exit(237);
        }
    }
    
    return path;
}



Channel::Channel common::finalState(const TString& filename)
{
    std::vector<Channel::Channel> v_channel {Channel::ee, Channel::emu, Channel::mumu};
    for(auto channel : v_channel){
        TString finalState(Channel::convert(channel));
        finalState.Prepend("/");
        finalState.Append("/");
        if(filename.Contains(finalState)){
            return channel;
        }
    }
    return Channel::undefined;
}



TString common::findFilelist(const TString& filelistDirectory,
                             const Channel::Channel& channel,
                             const Systematic::Systematic& systematic)
{
    TString result("");
    
    const TString filelistName(filelistDirectory + "/HistoFileList_" + systematic.name() + "_" + Channel::convert(channel) + ".txt");
    std::ifstream fileList(filelistName);
    if(!fileList.fail()){
        result = filelistName;
        fileList.close();
    }
    
    return result;
}



std::vector<Systematic::Systematic> common::findSystematicsFromFilelists(const TString& filelistDirectory,
                                                                         const std::vector<Channel::Channel>& v_channel,
                                                                         const std::vector<Systematic::Systematic>& v_systematic)
{
    std::vector<Systematic::Systematic> result;
    
    for(const auto& systematic : v_systematic){
        bool foundSystematic(true);
        for(const auto& channel : v_channel){
            const TString filelistName = findFilelist(filelistDirectory, channel, systematic);
            if(filelistName == ""){
                foundSystematic = false;
                break;
            }
        }
        if(foundSystematic) result.push_back(systematic);
    }
    
    return result;
}



std::vector<TString> common::readFilelist(const TString& filelistDirectory,
                                          const Channel::Channel& channel,
                                          const Systematic::Systematic& systematic,
                                          const std::vector<TString>& v_pattern)
{
    // Access name of file list containing list of input root files
    const TString filelistName = findFilelist(filelistDirectory, channel, systematic);
    if(filelistName == ""){
        std::cerr<<"Error in common::readFilelist()! Cannot find file (folder, channel, systematic): "
                 <<filelistDirectory<<" , "<<Channel::convert(channel)<<" , "<<systematic.name()<<"\n...break\n"<<std::endl;
        exit(1);
    }
    
    // Read in file list to a vector and return it
    return readFile(filelistName, v_pattern);
}



std::vector<TString> common::readFile(const TString& filename, const std::vector<TString>& v_pattern)
{
    std::vector<TString> v_line;
    
    std::cout<<"Reading file: "<<filename<<std::endl;
    std::ifstream file(filename);
    if(!file.is_open()){
        std::cerr<<"Error in common::readFile()! Cannot find file: "<<filename<<"\n...break\n"<<std::endl;
        exit(1);
    }
    while(!file.eof()){
        TString line;
        file>>line;
        // Skip empty lines
        if(line == "") continue;
        // Comment lines with '#'
        if(line.BeginsWith("#")) continue;
        // Check that all patterns are contained in the line
        bool notAllPatternsContained(false);
        for(const auto& pattern : v_pattern)
            if(!line.Contains(pattern)){
                notAllPatternsContained = true;
                break;
            }
        if(!notAllPatternsContained) v_line.push_back(line);
    }
    file.close();
    
    return v_line;
}






