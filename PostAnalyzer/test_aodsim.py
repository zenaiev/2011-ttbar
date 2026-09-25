import sys
import types
import struct

import numpy as np
import pandas as pd


# ============================================================
# uproot
# ============================================================

site = '/home/zenaiev/.local/lib/python3.14/site-packages'
sys.path.insert(0, site)

fake = types.ModuleType('awkward_cpp')


class DummyK:
  def __getitem__(self, key):
    return None


fake.cpu_kernels = types.SimpleNamespace(kernel=DummyK())
fake.libawkward = None

lib = types.ModuleType('awkward_cpp.lib')
ext = types.ModuleType('awkward_cpp.lib._ext')
lib._ext = ext

for name in ['ForthMachine32', 'ForthMachine64', 'ArrayBuilder']:
  setattr(ext, name, object)

sys.modules['awkward_cpp'] = fake
sys.modules['awkward_cpp.lib'] = lib
sys.modules['awkward_cpp.lib._ext'] = ext
sys.modules['cramjam'] = types.ModuleType('cramjam')

xxhash = types.ModuleType('xxhash')
xxhash.xxh64 = lambda *args, **kwargs: None
sys.modules['xxhash'] = xxhash

import uproot


# ============================================================
# Open ROOT file
# ============================================================

path = '../../0005D1FB-4BCF-E311-9FE4-002590A8312A.root'

f = uproot.open(path)
b = f['Events']['recoGenParticles_genParticles__SIM.obj']

# First basket = first event
data = bytes(b.basket(0).data)


# ============================================================
# GenParticle vector
# ============================================================

n = struct.unpack_from('>I', data, 8)[0]

offset = 12


# ============================================================
# Helpers
# ============================================================

def read_array(dtype):
  global offset

  a = np.frombuffer(
    data,
    dtype=dtype,
    count=n,
    offset=offset,
  ).copy()

  offset += a.dtype.itemsize*n

  return a


# ============================================================
# Kinematics
# ============================================================

qx3 = read_array('>i4')

pt = read_array('>f4').astype(np.float64)
eta = read_array('>f4').astype(np.float64)
phi = read_array('>f4').astype(np.float64)
mass = read_array('>f4').astype(np.float64)


# ============================================================
# PositionVector3D<Double32_t>
# ============================================================

offset += 32*n


# ============================================================
# Particle ID / status
# ============================================================

pdgId = read_array('>i4')
status = read_array('>i4')


# ============================================================
# edm::RefVector
# ============================================================

def read_refvector():
  global offset

  # ----------------------------------------------------------
  # edm::RefVector
  # ----------------------------------------------------------

  raw = struct.unpack_from('>I', data, offset)[0]
  offset += 4

  nbytes = raw & 0x3fffffff

  # ROOT byte count includes the 2-byte version.
  end = offset + nbytes

  version = struct.unpack_from('>H', data, offset)[0]
  offset += 2

  # ----------------------------------------------------------
  # edm::RefVectorBase
  # ----------------------------------------------------------

  raw = struct.unpack_from('>I', data, offset)[0]
  offset += 4

  base_nbytes = raw & 0x3fffffff

  # ROOT byte count includes the version.
  base_end = offset + base_nbytes

  base_version = struct.unpack_from('>H', data, offset)[0]
  offset += 2

  # ----------------------------------------------------------
  # edm::RefCore
  # ----------------------------------------------------------

  process_index, product_index, extra = struct.unpack_from(
    '>HHH',
    data,
    offset,
  )

  offset += 6

  # ----------------------------------------------------------
  # keys_ : vector<unsigned int>
  # ----------------------------------------------------------

  raw = struct.unpack_from('>I', data, offset)[0]
  offset += 4

  keys_nbytes = raw & 0x3fffffff

  # ROOT byte count includes the version.
  keys_end = offset + keys_nbytes

  keys_version = struct.unpack_from('>H', data, offset)[0]
  offset += 2

  count = struct.unpack_from('>I', data, offset)[0]
  offset += 4

  keys = np.frombuffer(
    data,
    dtype='>u4',
    count=count,
    offset=offset,
  ).astype(np.int64).tolist()

  offset += 4*count

  # ----------------------------------------------------------
  # Checks
  # ----------------------------------------------------------

  assert offset == keys_end, (
    'keys:',
    offset,
    keys_end,
    keys_nbytes,
  )

  assert offset == base_end, (
    'RefVectorBase:',
    offset,
    base_end,
    base_nbytes,
  )

  assert offset == end, (
    'RefVector:',
    offset,
    end,
    nbytes,
    base_nbytes,
    keys_nbytes,
  )

  return keys


# ============================================================
# Daughter / mother
# ============================================================

daughter = [
  read_refvector()
  for _ in range(n)
]

mother = [
  read_refvector()
  for _ in range(n)
]


# numberOfDaughters()
numberOfDaughters = np.array(
  [len(x) for x in daughter],
  dtype=np.int32,
)


# ============================================================
# collisionId
# ============================================================

collisionId = read_array('>i4')


# ============================================================
# Check complete basket
# ============================================================

assert offset == len(data), (
  f'Parser stopped at {offset}, '
  f'basket has {len(data)} bytes'
)


# ============================================================
# Four-momentum
# ============================================================

px = pt*np.cos(phi)
py = pt*np.sin(phi)

pz = np.full(n, np.nan)

mask = np.isfinite(eta) & (np.abs(eta) < 100)

pz[mask] = pt[mask]*np.sinh(eta[mask])

E = np.sqrt(
  px*px +
  py*py +
  np.where(np.isfinite(pz), pz*pz, 0) +
  mass*mass
)

E[~np.isfinite(pz)] = np.nan


# ============================================================
# DataFrame
# ============================================================

df = pd.DataFrame({
  'index': np.arange(n),

  'pdgId': pdgId,
  'status': status,

  'px': px,
  'py': py,
  'pz': pz,
  'E': E,

  'pt': pt,
  'eta': eta,
  'phi': phi,
  'mass': mass,

  'numberOfDaughters': numberOfDaughters,
  'daughter': daughter,
  'mother': mother,

  'collisionId': collisionId,
})


# ============================================================
# Print
# ============================================================

for i in range(n):
  print(
    f'{i:4d} '
    f'pdgId={pdgId[i]:6d} '
    f'status={status[i]:2d} '
    f'm={mass[i]:12.6f} '
    f'px={px[i]:12.6f} '
    f'py={py[i]:12.6f} '
    f'pz={pz[i]:12.6f} '
    f'E={E[i]:12.6f} '
    f'nDau={numberOfDaughters[i]:3d} '
    f'daughter={daughter[i]} '
    f'mother={mother[i]} '
    f'collisionId={collisionId[i]}'
  )


# ============================================================
# Save
# ============================================================

df.to_csv(
  'genparticles_first_event_full.csv',
  index=False,
  float_format='%.10g',
)