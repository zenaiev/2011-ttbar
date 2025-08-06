from glob import glob
import numpy as np

if __name__ == '__main__':
  message = ''
  testdir = 'kr_performance'
  refdir = 'ref_ev10000_ch3'
  files_ref = sorted(glob(f'{refdir}/*'))
  if len(files_ref) == 0:
    message += f'Reference directory {refdir} is empty'
  for fref in files_ref:
    f = fref.replace(refdir+'/', testdir+'/')
    data1 = np.loadtxt(fref)
    data2 = np.loadtxt(f)
    if data1.shape != data2.shape:
      message += f'{f} and {fref} have different shapes\n'
      continue
    result = np.allclose(data1, data2, equal_nan=True)
    if not result:
      message += f'{f} and {fref} are different\n'
  if message != '':
    print(f'\033[31mTEST FAILED\033[0m')
    print(message)
  else:
    print(f'\033[1;92mTEST PASSED\033[0m')
