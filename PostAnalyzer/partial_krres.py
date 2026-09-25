import numpy as np
import matplotlib.pyplot as plt
import uproot

vars = {
    'mtt': {'title': r'm(t\bar{t})', 'units': ' [GeV]', 'bins': np.concatenate((np.linspace(340, 450, 5, endpoint=False), np.logspace(np.log10(450), np.log10(700), 5, endpoint=False), np.logspace(np.log10(700), np.log10(1400), 8)))},
    'pttt': {'title': r'p_{T}(t\bar{t})', 'units': ' [GeV]', 'bins': np.concatenate((np.linspace(0, 100, 5, endpoint=False), np.logspace(np.log10(100), np.log10(250), 5, endpoint=False), np.logspace(np.log10(250), np.log10(800), 8)))},
    'ytt': {'title': r'y(t\bar{t})', 'units': '', 'bins': np.array([-2.4, -2.0, -1.75] + np.linspace(-1.6, 1.6, 16, endpoint=False).tolist() + [1.75, 2.0, 2.4])},
}
data = uproot.concatenate({f: 'ttbarTree' for f in [
        f'ntuples_full_output/ttbar_output_full_{ch}.root' for ch in range(1,4)
#        './ttbar_output_full_1.root','./ttbar_output_full_2.root'
    ]},
        ['lkrv3'] + [f'{var}_{suf}' for var in vars for suf in ['gen', 'lkrv3']] + [
         'mcNu', 'mcNubar','mcLp','mcLm','mcJ1','mcJ2','mcWp','mcWm',
         'mcT','mcTbar',
         'recoMetPx','recoMetPy','recoLm','recoLp','recoJ1','recoJ2',
        ],
        library='np'
    )

def p4(v):
    e = np.sqrt(np.sum(v**2, axis=1))
    ret = np.copy(v)
    ret[:,3] = e
    return ret

def solve_lkrv3(lp, lm, met_x, met_y, j1, j2):
    ll = lp + lm
    nn = np.zeros(shape=lp.shape)
    nn[:,0] = met_x
    nn[:,1] = met_y
    nunu_pt = np.sqrt(met_x**2 + met_y**2)
    ll_m2 = ll[:,3]**2 - ll[:,0]**2 - ll[:,1]**2 - ll[:,2]**2
    assert np.min(ll_m2) > 0
    ll_m = np.sqrt(ll_m2)
    nunu_mt = np.sqrt(ll_m**2 + nunu_pt**2)
    ll_y = 0.5 * np.log((ll[:,3] + ll[:,2]) / (ll[:,3] - ll[:,2]))
    nn[:,2] = nunu_mt * np.sinh(ll_y)
    nn[:,3] = nunu_mt * np.cosh(ll_y)
    assert np.all(nn[:,3]**2 - nn[:,0]**2 - nn[:,1]**2 - nn[:,2]**2) > -0.001
    mw = 80.4
    llnn = ll + nn
    llnn_m = np.sqrt(llnn[:,3]**2 - llnn[:,0]**2 - llnn[:,1]**2 - llnn[:,2]**2)
    llnn_y = 0.5 * np.log((llnn[:,3] + llnn[:,2]) / (llnn[:,3] - llnn[:,2]))
    enew = np.sqrt(4*mw**2 + llnn[:,0]**2 + llnn[:,1]**2) * np.cosh(llnn_y)
    znew = enew * np.tanh(llnn_y)
    mask = llnn_m < 2 * mw
    llnn[mask,2] = znew[mask]
    llnn[mask,3] = enew[mask]
    return llnn + j1 + j2

class KR():
    def mtt(self, data, mask):
        self.rec(data, mask)
        m2 = self.e**2 - self.x**2 - self.y**2 - self.z**2
        return np.sqrt(np.maximum(m2, 0))
    def pttt(self, data, mask):
        self.rec(data, mask)
        return np.sqrt(self.x**2 + self.y**2)
    def ytt(self, data, mask):
        self.rec(data, mask)
        return 0.5 * np.log((self.e + self.z) / (self.e - self.z))

    
class LKRv3(KR):
    def __init__(self):
        self.label = 'LKRv3'
    def mtt(self, data, mask):
        return data['mtt_lkrv3'][mask]
    def pttt(self, data, mask):
        return data['pttt_lkrv3'][mask]
    def ytt(self, data, mask):
        return data['ytt_lkrv3'][mask]

class LKRv3CHECK(KR):
    def __init__(self):
        self.label = 'LKRv3CHECK'
    def rec(self, data, mask):
        ttbar = solve_lkrv3(p4(data['recoLp'][mask]), p4(data['recoLm'][mask]), data['recoMetPx'][mask], data['recoMetPy'][mask], p4(data['recoJ1'][mask]), p4(data['recoJ2'][mask]))
        self.x = ttbar[:,0]
        self.y = ttbar[:,1]
        self.z = ttbar[:,2]
        self.e = ttbar[:,3]

class LKRv3noResLeptons(KR):
    def __init__(self):
        self.label = 'LKRv3 no res leptons'
    def rec(self, data, mask):
        ttbar = solve_lkrv3(p4(data['mcLp'][mask]), p4(data['mcLm'][mask]), data['recoMetPx'][mask], data['recoMetPy'][mask], p4(data['recoJ1'][mask]), p4(data['recoJ2'][mask]))
        self.x = ttbar[:,0]
        self.y = ttbar[:,1]
        self.z = ttbar[:,2]
        self.e = ttbar[:,3]

class LKRv3noResJets(KR):
    def __init__(self):
        self.label = 'LKRv3 no res jets'
    def rec(self, data, mask):
        ttbar = solve_lkrv3(p4(data['recoLp'][mask]), p4(data['recoLm'][mask]), data['recoMetPx'][mask], data['recoMetPy'][mask], p4(data['mcJ1'][mask]), p4(data['mcJ2'][mask]))
        self.x = ttbar[:,0]
        self.y = ttbar[:,1]
        self.z = ttbar[:,2]
        self.e = ttbar[:,3]

class LKRv3noResMET(KR):
    def __init__(self):
        self.label = 'LKRv3 no res met'
    def rec(self, data, mask):
        ttbar = solve_lkrv3(p4(data['recoLp'][mask]), p4(data['recoLm'][mask]), data['mcNu'][mask][:,0] + data['mcNubar'][mask][:,0], data['mcNu'][mask][:,1] + data['mcNubar'][mask][:,1], p4(data['recoJ1'][mask]), p4(data['recoJ2'][mask]))
        self.x = ttbar[:,0]
        self.y = ttbar[:,1]
        self.z = ttbar[:,2]
        self.e = ttbar[:,3]

class KRsolve(KR):
    def __init__(self):
        self.label = 'solve'
    def rec(self, data, mask):
        ttbar = solve_lkrv3(p4(data['mcLp'][mask]), p4(data['mcLm'][mask]), data['mcNu'][mask][:,0] + data['mcNubar'][mask][:,0], data['mcNu'][mask][:,1] + data['mcNubar'][mask][:,1], p4(data['mcJ1'][mask]), p4(data['mcJ2'][mask]))
        self.x = ttbar[:,0]
        self.y = ttbar[:,1]
        self.z = ttbar[:,2]
        self.e = ttbar[:,3]

class KRsolveDetJets(KR):
    def __init__(self):
        self.label = 'solve + det. jets'
    def rec(self, data, mask):
        ttbar = solve_lkrv3(p4(data['mcLp'][mask]), p4(data['mcLm'][mask]), data['mcNu'][mask][:,0] + data['mcNubar'][mask][:,0], data['mcNu'][mask][:,1] + data['mcNubar'][mask][:,1], p4(data['recoJ1'][mask]), p4(data['recoJ2'][mask]))
        self.x = ttbar[:,0]
        self.y = ttbar[:,1]
        self.z = ttbar[:,2]
        self.e = ttbar[:,3]

class KRsolveDetMET(KR):
    def __init__(self):
        self.label = 'solve + det. MET'
    def rec(self, data, mask):
        ttbar = solve_lkrv3(p4(data['mcLp'][mask]), p4(data['mcLm'][mask]), data['recoMetPx'][mask], data['recoMetPy'][mask], p4(data['mcJ1'][mask]), p4(data['mcJ2'][mask]))
        self.x = ttbar[:,0]
        self.y = ttbar[:,1]
        self.z = ttbar[:,2]
        self.e = ttbar[:,3]

class KRsolveDetLeptons(KR):
    def __init__(self):
        self.label = 'solve + det. leptons'
    def rec(self, data, mask):
        ttbar = solve_lkrv3(p4(data['recoLp'][mask]), p4(data['recoLm'][mask]), data['mcNu'][mask][:,0] + data['mcNubar'][mask][:,0], data['mcNu'][mask][:,1] + data['mcNubar'][mask][:,1], p4(data['mcJ1'][mask]), p4(data['mcJ2'][mask]))
        self.x = ttbar[:,0]
        self.y = ttbar[:,1]
        self.z = ttbar[:,2]
        self.e = ttbar[:,3]

class KRReso(KR):
    def __init__(self):
        self.label = 'resolution'
    def rec(self, data, mask):
        t = p4(data['mcT'][mask])
        tbar = p4(data['mcTbar'][mask])
        met = np.zeros(shape=t.shape)
        met[:,0] = data['recoMetPx'][mask]
        met[:,1] = data['recoMetPy'][mask]
        nunubarpt = np.zeros(shape=t.shape)
        nunubarpt[:,0] = data['mcNu'][mask][:,0] + data['mcNubar'][mask][:,0]
        nunubarpt[:,1] = data['mcNu'][mask][:,1] + data['mcNubar'][mask][:,1]
        lp = p4(data['mcLp'][mask])
        lm = p4(data['mcLm'][mask])
        recoLp = p4(data['recoLp'][mask])
        recoLm = p4(data['recoLm'][mask])
        j1 = p4(data['mcJ1'][mask])
        j2 = p4(data['mcJ2'][mask])
        recoJ1 = p4(data['recoJ1'][mask])
        recoJ2 = p4(data['recoJ2'][mask])
        self.reco = t+tbar-nunubarpt-lp-lm-j1-j2+met+recoLp+recoLm+recoJ1+recoJ2
        self.x = self.reco[:,0]
        self.y = self.reco[:,1]
        self.z = self.reco[:,2]
        self.e = self.reco[:,3]

class KRMET(KR):
    def __init__(self):
        self.label = 'MET'
    def rec(self, data, mask):
        t = p4(data['mcT'][mask])
        tbar = p4(data['mcTbar'][mask])
        met = np.zeros(shape=t.shape)
        met[:,0] = data['recoMetPx'][mask]
        met[:,1] = data['recoMetPy'][mask]
        nunubarpt = np.zeros(shape=t.shape)
        nunubarpt[:,0] = data['mcNu'][mask][:,0] + data['mcNubar'][mask][:,0]
        nunubarpt[:,1] = data['mcNu'][mask][:,1] + data['mcNubar'][mask][:,1]
        self.reco = t+tbar-nunubarpt+met
        self.x = self.reco[:,0]
        self.y = self.reco[:,1]
        self.z = self.reco[:,2]
        self.e = self.reco[:,3]

class KRLEP(KR):
    def __init__(self):
        self.label = 'leptons'
    def rec(self, data, mask):
        t = p4(data['mcT'][mask])
        tbar = p4(data['mcTbar'][mask])
        lp = p4(data['mcLp'][mask])
        lm = p4(data['mcLm'][mask])
        recoLp = p4(data['recoLp'][mask])
        recoLm = p4(data['recoLm'][mask])
        self.reco = t+tbar-lp-lm+recoLp+recoLm
        self.x = self.reco[:,0]
        self.y = self.reco[:,1]
        self.z = self.reco[:,2]
        self.e = self.reco[:,3]

class KRJETS(KR):
    def __init__(self):
        self.label = 'jets'
    def rec(self, data, mask):
        t = p4(data['mcT'][mask])
        tbar = p4(data['mcTbar'][mask])
        j1 = p4(data['mcJ1'][mask])
        j2 = p4(data['mcJ2'][mask])
        recoJ1 = p4(data['recoJ1'][mask])
        recoJ2 = p4(data['recoJ2'][mask])
        self.reco = t+tbar-j1-j2+recoJ1+recoJ2
        self.x = self.reco[:,0]
        self.y = self.reco[:,1]
        self.z = self.reco[:,2]
        self.e = self.reco[:,3]

class KRJETSMET(KR):
    def __init__(self):
        self.label = 'MET+jets'
    def rec(self, data, mask):
        t = p4(data['mcT'][mask])
        tbar = p4(data['mcTbar'][mask])
        j1 = p4(data['mcJ1'][mask])
        j2 = p4(data['mcJ2'][mask])
        recoJ1 = p4(data['recoJ1'][mask])
        recoJ2 = p4(data['recoJ2'][mask])
        met = np.zeros(shape=t.shape)
        met[:,0] = data['recoMetPx'][mask]
        met[:,1] = data['recoMetPy'][mask]
        nunubarpt = np.zeros(shape=t.shape)
        nunubarpt[:,0] = data['mcNu'][mask][:,0] + data['mcNubar'][mask][:,0]
        nunubarpt[:,1] = data['mcNu'][mask][:,1] + data['mcNubar'][mask][:,1]
        self.reco = t+tbar-j1-j2+recoJ1+recoJ2-nunubarpt+met
        self.x = self.reco[:,0]
        self.y = self.reco[:,1]
        self.z = self.reco[:,2]
        self.e = self.reco[:,3]

class KRmcgen6(KR):
    def __init__(self):
        self.label = 'MC gen. l+lbar+nu+nubar+b+bbar'
    def rec(self, data, mask):
        j1 = p4(data['mcJ1'][mask])
        j2 = p4(data['mcJ2'][mask])
        lp = p4(data['mcLp'][mask])
        lm = p4(data['mcLm'][mask])
        nu = p4(data['mcNu'][mask])
        nubar = p4(data['mcNubar'][mask])
        self.reco = j1+j2+lp+lm+nu+nubar
        self.x = self.reco[:,0]
        self.y = self.reco[:,1]
        self.z = self.reco[:,2]
        self.e = self.reco[:,3]

class KRmcgennunubarze(KR):
    def __init__(self):
        self.label = 'MC gen. nu+nubar(pZ,E)'
    def rec(self, data, mask):
        j1 = p4(data['mcJ1'][mask])
        j2 = p4(data['mcJ2'][mask])
        lp = p4(data['mcLp'][mask])
        lm = p4(data['mcLm'][mask])
        nu = p4(data['mcNu'][mask])
        nubar = p4(data['mcNubar'][mask])

        j1 = p4(data['mcJ1'][mask])
        mt = np.sqrt(data['mttt_lkrv3'][mask]**2+data['pttt_lkrv3'][mask]**2)
        self.e = mt*np.cosh(data['ytt_lkrv3'][mask])
        self.z = mt*np.sinh(data['ytt_lkrv3'][mask])
        #self.x = 
        j1 = p4(data['mcJ1'][mask])
        j2 = p4(data['mcJ2'][mask])
        lp = p4(data['mcLp'][mask])
        lm = p4(data['mcLm'][mask])
        nu = p4(data['mcNu'][mask])
        nubar = p4(data['mcNubar'][mask])
        self.reco = j1+j2+lp+lm+nu+nubar
        self.x = self.reco[:,0]
        self.y = self.reco[:,1]
        self.z = self.reco[:,2]
        self.e = self.reco[:,3]

#krs = [LKRv3(), KRmcgen6(), KRMET(), KRLEP(), KRJETS(), KRJETSMET()]
#krs = [LKRv3(), LKRv3CHECK()]
#krs = [LKRv3CHECK(), KRmcgen6(), KRsolve(), KRJETS(), KRMET(), KRJETSMET(), KRLEP()]
#krs = [LKRv3CHECK(), KRmcgen6(), KRsolve(), LKRv3noResJets(), LKRv3noResMET(), LKRv3noResLeptons()]
#krs = [LKRv3CHECK(), KRmcgen6(), KRsolve(), KRsolveDetJets(), KRsolveDetMET(), KRsolveDetLeptons()]
krs = [LKRv3CHECK(), KRsolve(), KRsolveDetJets(), KRsolveDetMET(), KRsolveDetLeptons()]

fig, ax = plt.subplots(2, 3, figsize=(9, 6))
#fig_h, ax_h = plt.subplots(1, 1, figsize=(9, 6))
for ivar,var in enumerate(vars.keys()):
  mask = (data['lkrv3'] == 1) & (data[f'{var}_lkrv3'] > -9999.) & (data[f'{var}_lkrv3'] < 9999.)
  var_gen = data[f'{var}_gen'][mask]
  diffs = [getattr(kr, var)(data, mask) - var_gen for kr in krs]
  rmss = [[] for _ in krs]
  centers = []
  for lo, hi in zip(vars[var]['bins'][:-1], vars[var]['bins'][1:]):
      mask_bin = (var_gen >= lo) & (var_gen < hi)
      centers.append((lo + hi) / 2)
      ref = None
      for ikr,kr in enumerate(krs):
        d = diffs[ikr][mask_bin]
        '''if ikr == 0:
            ref = d
        else:
            #print(d.shape, ref.shape)
            mismatch = d - ref
            #print(np.min(mismatch), np.max(mismatch))
            if lo == 0:
                print(mismatch)
                ax_h.hist(mismatch, np.arange(-1000,1000,1))'''
        #print(lo, hi, d)
        #assert len(d) > 0
        rmss[ikr].append(np.sqrt(np.mean(d**2)))
  #ax_h.set_yscale('log')
  #fig_h.savefig('tmp.pdf')
  #print(rmss)
  rmss.append([])
  for ib,_ in enumerate(vars[var]['bins'][1:]):
    for ikr,kr in enumerate(krs):
      if ikr > 1:
        tmp2 = np.maximum(rmss[ikr][ib]**2 - rmss[1][ib]**2, 0)
        rmss[ikr][ib] = np.sqrt(tmp2)
    rmss[len(krs)].append(np.sqrt(sum(rmss[ikr][ib]**2 for ikr in range(1, len(krs)))))
  for ikr,kr in enumerate(krs):
    ax[0, ivar].plot(centers, rmss[ikr], 'o-', label=kr.label)
    ax[1, ivar].plot(centers, np.array(rmss[ikr]) / np.array(rmss[0]), 'o-', label=kr.label)
    #ax[0, ivar].plot(centers, np.array(rmss[ikr])/np.array(rmss[1]), 'o-', label=kr.label)
    print(var, [float(f'{v:.1f}') for v in rmss[ikr]], kr.label)
  ax[0, ivar].plot(centers, rmss[len(krs)], 'o-', label='sum')
  ax[1, ivar].plot(centers, np.array(rmss[len(krs)]) / np.array(rmss[0]), 'o-', label='sum')
  ax[0, ivar].set_xlabel(f'${vars[var]["title"]}^{{gen}}${vars[var]["units"]}')
  ax[0, ivar].set_ylabel(f'RMS {vars[var]["units"]}')
  ax[0, ivar].set_ylim(0.,None)
  ax[0, ivar].grid(True, alpha=0.3)
ax[0, 0].legend()
fig.tight_layout()
fig.savefig('partial_krres_mtt.pdf')
