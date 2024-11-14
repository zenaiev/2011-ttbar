module () {
        eval `/usr/bin/modulecmd zsh $*`
}
module use -a /afs/desy.de/group/cms/modulefiles/
module load cmssw
cmssw-slc6 --command-to-run "cd /nfs/dust/cms/user/zenaiev/od/20241112-slc6/CMSSW_5_3_32/src/2011-ttbar/Analyzer && cmsenv && cmsRun analyzer_cfg.py"
