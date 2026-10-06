import ROOT
import numpy as np
import matplotlib.pyplot as plt

f = ROOT.TFile.Open('response_matrix_chi2.root')
# x = true, y = reco

labels = ['resp_lkr', 'resp_lkrnn', 'resp_lkrnnfreeraw', 'resp_lkrnnmlpl1zmlj', 'resp_lkrnnflowmlj']
fig,axs = plt.subplots(2, len(labels), figsize=(12,6), sharex=True, sharey='row')
for ilabel,label in enumerate(labels):
  #h = f.Get('resp_lkr')
  #h = f.Get('resp_lkrnn')
  #h = f.Get('resp_lkrnnfreeraw')
  #h = f.Get('resp_lkrnnmlpl1zmlj')
  #h = f.Get('resp_lkrnnflowmlj')
  h = f.Get(label)
  h_true = f.Get('truth')

  nx = h.GetNbinsX()
  ny = h.GetNbinsY()

  arr = np.array([[h.GetBinContent(ix, iy) for iy in range(1, ny + 1)] for ix in range(1, nx + 1)])
  arr_norm = arr / arr.sum(axis=1, keepdims=True)
  xedges = np.array([h.GetXaxis().GetBinLowEdge(i) for i in range(1, h.GetNbinsX() + 1)] + [h.GetXaxis().GetBinUpEdge(h.GetNbinsX())])
  yedges = np.array([h.GetYaxis().GetBinLowEdge(i) for i in range(1, h.GetNbinsY() + 1)] + [h.GetYaxis().GetBinUpEdge(h.GetNbinsY())])
  truth = np.array([h_true.GetBinContent(ix) for ix in range(1, nx + 1)])

  #print(arr.shape)
  #print(arr)
  #print(xedges.shape)
  #print(xedges)
  #assert np.all(xedges == yedges)
  #print(np.sum(arr), np.sum(truth))

  proj_x = arr.sum(axis=1)
  proj_y = arr.sum(axis=0)
  pred_y = arr_norm.T @ proj_x

  br = 0.046
  lumi = 2500.0
  #xsec13 = 850.
  xsec = 165.6
  #w_data = lumi / 54990752. * xsec * br
  excess1 = 6.49 / 4.
  excess = np.zeros_like(proj_x)
  excess[0] = excess1
  #print(proj_x_topo / proj_x)

  pred_x = proj_x / np.sum(proj_x) * lumi * xsec * br
  pred_y = arr_norm.T @ pred_x
  pred_x_topo = (pred_x / np.sum(pred_x) * xsec + excess)  * lumi * br
  pred_y_topo = arr_norm.T @ pred_x_topo
  #print(pred_y - proj_y)
  ratio = (pred_y/np.sum(pred_y)) / (proj_y/np.sum(proj_y))
  #print(ratio)
  assert np.all(np.abs(ratio - 1) < 1e-10)
  #print(pred_y * w_data)
  #print((pred_y_topo - pred_y) * w_data)
  chi2 = np.sum(((pred_y_topo - pred_y))**2 / (pred_y))
  sigma = np.sqrt(chi2)
  print(f'{label}: chi2 = {chi2:.3}, sigma = {sigma:.3}')
  axs[0, ilabel].stairs(pred_y, xedges, label = 'MC', color = 'k')
  axs[0, ilabel].stairs(pred_y_topo, xedges, label = 'MC + toponium', color = 'b')
  err = np.sqrt(pred_y)
  #axs[0, ilabel].fill_between(xedges, np.r_[pred_y - err, pred_y[-1] - err[-1]], np.r_[pred_y + err, pred_y[-1] + err[-1]], step='post', alpha=0.3, color = 'gray')
  axs[1, ilabel].stairs(pred_y_topo / pred_y_topo, xedges, color = 'k')
  axs[1, ilabel].stairs(pred_y_topo / pred_y, xedges, color = 'b')
  rat_err = np.sqrt(pred_y) / (pred_y)
  axs[1, ilabel].fill_between(xedges, np.r_[1 - rat_err, 1 - rat_err[-1]], np.r_[1 + rat_err, 1 + rat_err[-1]], step='post', alpha=0.3, color = 'gray')
  axs[1, ilabel].set_xlabel(r'$M(t\bar{t})$ [GeV]')
  if ilabel == 0:
    axs[0, ilabel].set_ylabel('Expected events')
    axs[1, ilabel].set_ylabel('Ratio to MC')
  axs[0, ilabel].set_xlim(340, 450)
  axs[0, ilabel].set_ylim(0, 250)
  axs[0, ilabel].set_title(label.split('_')[-1])
  axs[1, ilabel].set_ylim(0.5, 1.5)
  axs[1, ilabel].legend(title = fr'$\chi^{2} = {chi2:.2f}$ $({sigma:.2f}\sigma)$')
axs[0, -1].legend(title = r'Dilepton $t\bar{t}$, $\sqrt{s} = 7$ TeV')
fig.tight_layout()
fig.savefig('plots/toponium.pdf')
fig.savefig('plots/toponium.png')
aaa

fig,ax = plt.subplots(1, 2, figsize=(12,6))
ax[0].stairs(proj_x, xedges, label = 'proj_x')
ax[0].set_xlabel('X')
ax[0].set_ylabel('Integral over Y')
ax[1].stairs(proj_y * w_data, yedges, label = 'proj_y')
ax[1].stairs(pred_y * w_data, yedges, label = 'pred_y')
ax[1].stairs(pred_y_topo * w_data, yedges, label = 'pred_y_topo')
ax[1].legend()
ax[1].set_xlabel('Y')
ax[1].set_ylabel('Integral over X')

fig,ax = plt.subplots()
ax.stairs(arr[0, :], yedges)
ax.set_xlabel('Reco')
ax.set_ylabel('First true bin')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(300, 500)

#plt.grid()
plt.show()
