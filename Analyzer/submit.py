import os
import subprocess
import shutil
import argparse
import ROOT

def is_done(torun, events_proc):
  if 'events' in torun:
    if torun['events'] == events_proc:
      done = ' -> DONE'
    else:
      done = f' ... {events_proc/torun["events"]*100:.2f}%'
  else:
    done = ''
  return done

def submit(n1, n2=None):
  com = f'bash {run1_name} {torun["inlist"]} {name}/{torun["outdir"]} {torun["reco"]} {torun["gen"]} {torun["mc"]}'
  if type(n1) is int and type(n2) is int:
    com_submit = f'cs -nod-{torun["jobName"]}-{name} -o{basedir}/{name}/{torun["outdir"]}/log -a{n1}:{n2} {com}'
  else:
    assert type(n1) is list and n2 is None
    if len(n1) == 1:
      com_submit = f'cs -nod-{torun["jobName"]}-{name} -o{basedir}/{name}/{torun["outdir"]}/log-{n1[0]} {com} {n1[0]}'
    else:
      com_submit = f'cs -nod-{torun["jobName"]}-{name} -o{basedir}/{name}/{torun["outdir"]}/log -a{",".join(str(j) for j in n1)} {com}'
  if args.pretend:
    print(com_submit)
  else:
    subprocess.call(com_submit.split())

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='submit od jobs')
  parser.add_argument('-m', '--merge', type=str, help='merge ROOT files')
  parser.add_argument('-r', '--resubmit', action='store_true', help='resubmit failed jobs')
  parser.add_argument('-p', '--pretend', action='store_true', help='pretend (only print submit commands)')
  parser.add_argument('-o', '--overwrite', action='store_true', help='overwrite output directories')
  parser.add_argument('-s', '--samples', type=str, help='samples to run [s1,s2,...]')
  parser.add_argument('-n', '--name', type=str, default='20241112', help='run name')
  args = parser.parse_args()
  if args.merge and not args.resubmit:
    print('ERROR: --merge requires --resubmit')
    aaa
  name = args.name
  basedir = '/nfs/dust/cms/user/zenaiev/od/20241112-slc6/CMSSW_5_3_32/src/2011-ttbar/Analyzer'
  toruns = [
    {'inlist': 'data/CMS_Run2011A_MuEG_AOD_12Oct2013-v1-all_file_index.txt', 'outdir': 'ntuples-data/MuEG', 'jobName': 'tt-em', 'reco': 1, 'gen': 0, 'mc': 0, 'events': 33131255},
    {'inlist': 'data/CMS_Run2011A_DoubleMu_AOD_12Oct2013-v1-all_file_index.txt', 'outdir': 'ntuples-data/DoubleMu', 'jobName': 'tt-mm', 'reco': 1, 'gen': 0, 'mc': 0, 'events': 39668813},
    {'inlist': 'data/CMS_Run2011A_DoubleElectron_AOD_12Oct2013-v1-all_file_index.txt', 'outdir': 'ntuples-data/DoubleElectron', 'jobName': 'tt-ee', 'reco': 1, 'gen': 0, 'mc': 0, 'events': 49693737},
    {'inlist': 'mc/TTJets_TuneZ2_7TeV-madgraph-tauola/CMS_MonteCarlo2011_Summer11LegDR_TTJets_TuneZ2_7TeV-madgraph-tauola_AODSIM_PU_S13_START53_LV6-v1_00000_file_index.txt', 'outdir': 'ntuples-mc/TTJets_TuneZ2_7TeV-madgraph-tauola/00000', 'jobName': 'tt-mg1', 'reco': 1, 'gen': 1, 'mc': 1, 'events': 54990752},
    {'inlist': 'mc/TTJets_TuneZ2_7TeV-madgraph-tauola/CMS_MonteCarlo2011_Summer11LegDR_TTJets_TuneZ2_7TeV-madgraph-tauola_AODSIM_PU_S13_START53_LV6-v1_00001_file_index.txt', 'outdir': 'ntuples-mc/TTJets_TuneZ2_7TeV-madgraph-tauola/00001', 'jobName': 'tt-mg2', 'reco': 1, 'gen': 1, 'mc': 1, 'events': 54990752},
    {'inlist': 'mc/TTJets_TuneZ2_7TeV-madgraph-tauola/CMS_MonteCarlo2011_Summer11LegDR_TTJets_TuneZ2_7TeV-madgraph-tauola_AODSIM_PU_S13_START53_LV6-v1_010000_file_index.txt', 'outdir': 'ntuples-mc/TTJets_TuneZ2_7TeV-madgraph-tauola/010000', 'jobName': 'tt-mg3', 'reco': 1, 'gen': 1, 'mc': 1, 'events': 54990752},
    {'inlist': 'mc/TTJets_TuneZ2_7TeV-madgraph-tauola/CMS_MonteCarlo2011_Summer11LegDR_TTJets_TuneZ2_7TeV-madgraph-tauola_AODSIM_PU_S13_START53_LV6-v1_010001_file_index.txt', 'outdir': 'ntuples-mc/TTJets_TuneZ2_7TeV-madgraph-tauola/010001', 'jobName': 'tt-mg4', 'reco': 1, 'gen': 1, 'mc': 1, 'events': 54990752},
    {'inlist': 'mc/TTJets_TuneZ2_7TeV-madgraph-tauola/CMS_MonteCarlo2011_Summer11LegDR_TTJets_TuneZ2_7TeV-madgraph-tauola_AODSIM_PU_S13_START53_LV6-v1_010002_file_index.txt', 'outdir': 'ntuples-mc/TTJets_TuneZ2_7TeV-madgraph-tauola/010002', 'jobName': 'tt-mg5', 'reco': 1, 'gen': 1, 'mc': 1, 'events': 54990752},
    {'inlist': 'mc/TTJets_TuneZ2_7TeV-madgraph-tauola/CMS_MonteCarlo2011_Summer11LegDR_TTJets_TuneZ2_7TeV-madgraph-tauola_AODSIM_PU_S13_START53_LV6-v1_010003_file_index.txt', 'outdir': 'ntuples-mc/TTJets_TuneZ2_7TeV-madgraph-tauola/010003', 'jobName': 'tt-mg6', 'reco': 1, 'gen': 1, 'mc': 1, 'events': 54990752},
    {'inlist': 'mc/CMS_MonteCarlo2011_Summer11LegDR_T_TuneZ2_tW-channel-DR_7TeV-powheg-tauola_AODSIM_PU_S13_START53_LV6_file_index.txt', 'outdir': 'ntuples-mc/T_TuneZ2_tW-channel-DR_7TeV-powheg-tauola', 'jobName': 'tt-st', 'reco': 1, 'gen': 0, 'mc': 1, 'events': 744859},
    {'inlist': 'mc/CMS_MonteCarlo2011_Summer11LegDR_Tbar_TuneZ2_tW-channel-DR_7TeV-powheg-tauola_AODSIM_PU_S13_START53_LV6-v1_00000_file_index.txt', 'outdir': 'ntuples-mc/Tbar_TuneZ2_tW-channel-DR_7TeV-powheg-tauola', 'jobName': 'tt-stbar', 'reco': 1, 'gen': 0, 'mc': 1, 'events': 801626},
    {'inlist': 'mc/CMS_MonteCarlo2011_Summer11LegDR_WJetsToLNu_TuneZ2_7TeV-madgraph-tauola_AODSIM_PU_S13_START53_LV6_file_index.txt', 'outdir': 'ntuples-mc/WJetsToLNu_TuneZ2_7TeV-madgraph-tauola', 'jobName': 'tt-wj', 'reco': 1, 'gen': 0, 'mc': 1, 'events': 78347691},
    {'inlist': 'mc/CMS_MonteCarlo2011_Summer11LegDR_DYJetsToLL_M-10To50_TuneZ2_7TeV-pythia6_AODSIM_PU_S13_START53_LV6-v1_00000_file_index.txt', 'outdir': 'ntuples-mc/DYJetsToLL_M-10To50_TuneZ2_7TeV-pythia6', 'jobName': 'tt-dyl', 'reco': 1, 'gen': 0, 'mc': 1, 'events': 39909640},
    {'inlist': 'mc/CMS_MonteCarlo2011_Summer11LegDR_DYJetsToLL_TuneZ2_M-50_7TeV-madgraph-tauola_AODSIM_PU_S13_START53_LV6_file_index.txt', 'outdir': 'ntuples-mc/DYJetsToLL_TuneZ2_M-50_7TeV-madgraph-tauola', 'jobName': 'tt-dyh', 'reco': 1, 'gen': 0, 'mc': 1, 'events': 36408225},
  ]
  n_resub = 0
  events_proc_mg = 0
  for torun in toruns:
    if args.samples and all(s not in torun['jobName'] for s in args.samples.split(',')):
      continue
    with open(f'{basedir}/{torun["inlist"]}') as f:
      nl = len(f.readlines())
    outdir = f'{basedir}/{name}/{torun["outdir"]}'
    run1_name = f'{outdir}/run1.sh'
    if args.resubmit:
      events_proc = 0
      events_sel = 0
      toresub = []
      jobs_running = 0
      jobs_complete = 0
      for j in range(1,nl+1):
        logname = f'{outdir}/log-{j}'
        if any(not os.path.exists(logname+ext) for ext in ['.txt', '.err', '.log']):
          continue
        with open(logname+'.txt') as f:
          ls = f.readlines()
          if len(ls)>=2 and all(pat in ls[-2] for pat in ['Processed', 'selected']):
            events_proc += int(ls[-2].split()[1])
            events_sel += int(ls[-2].split()[4])
            jobs_complete += 1
            continue
        with open(logname+'.log') as f:
          ls = f.readlines()
          if len(ls) < 2 or 'Job terminated of its own accord' not in ls[-2]:
            jobs_running += 1
            continue
        with open(logname+'.err') as f:
          if any ('container creation failed' in l for l in f.readlines()):
            toresub.append(j)
      done = ''
      if 'mg' in torun['jobName']:
        events_proc_mg += events_proc
        done = is_done(torun, events_proc_mg)
      else:
        done = is_done(torun, events_proc)
      print(f'{torun["jobName"]}: {nl} jobs, events processed[sel] {events_proc}[{events_sel}], {jobs_complete}[{jobs_complete/nl*100:.0f}%] complete, {jobs_running}[{jobs_running/nl*100:.0f}%] running, {len(toresub)}[{len(toresub)/nl*100:.0f}%] to be resubmitted{done}')
      if len(toresub) > 0:
        submit(toresub)
        n_resub += len(toresub)
      if args.merge:
        outdir_merge = f'{basedir}/{args.merge}/{torun["outdir"]}'
        if os.path.exists(outdir_merge):
          shutil.rmtree(outdir_merge)
        os.makedirs(outdir_merge)
        ch = ROOT.TChain('tree')
        for j in range(1, nl+1):
          ch.Add(f'{outdir}/ttbarSel_{j}.root')
        outfile_merged = ROOT.TFile(f'{outdir_merge}/ttbarSel_merged.root', 'recreate')
        mergedTree = ch.CloneTree()
        mergedTree.Write()
        outfile_merged.Close()
    else:
      if os.path.exists(outdir):
        if args.overwrite:
          shutil.rmtree(outdir)
        else:
          print(f'ERROR: outdir exists {outdir}')
          continue
      os.makedirs(outdir)
      with open(run1_name, 'w') as f:
        f.write(f'''
echo "Run1 script args: $@"
date
cd {basedir}
i=$6
head -n$i $1 | tail -n1 > $2/inputList_${{i}}.txt
module () {{
        eval `/usr/bin/modulecmd zsh $*`
}}
module use -a /afs/desy.de/group/cms/modulefiles/
module load cmssw
cmssw-slc6 --command-to-run "cmsenv && cmsRun analyzer_cfg.py $2/inputList_${{i}}.txt $2/ttbarSel_${{i}}.root $3 $4 $5"
date
''')
        submit(1, nl)
  if args.resubmit and n_resub > 0:
    print(f'Total resubmitted: {n_resub}')
  