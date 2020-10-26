import os, re
import h5py
import numpy as np
from os.path import join
from pathlib import Path
from tqdm import tqdm

PTH = '/scratch/richardso21/20-21_BGSUB/FgSegNet_O'
SITES = ['prudhoe_12', 'prudhoe_15', 'prudhoe_22']

# for each site
for site in SITES:
  print(site)
  # create h5py file
  OPTH = join(PTH, f'{site}.h5')
  f = h5py.File(OPTH, 'w')
  
  # collect pos/neg .npy files
  pos = [str(x) for x in Path(PTH).rglob(f'{site}_pos*.npy')]
  neg = [str(x) for x in Path(PTH).rglob(f'{site}_neg*.npy')]

  res = []
  # for each .npy
  for i in tqdm(sorted(pos, key=lambda f: int(re.sub('\D', '', f))), desc='POS'):
    # append to `res`
    res.append(np.load(i))
  # `np.vstack()` to concatenate them vertically
  res = np.vstack(res)
  # add to h5 dataset file
  dset_pos = f.create_dataset(f'POS', data=res)
  dset_pos.attrs['shape'] = res.shape
  
  # same for negatives
  res = []
  for i in tqdm(sorted(neg, key=lambda f: int(re.sub('\D', '', f))), desc='NEG'):
    res.append(np.load(i))
  res = np.vstack(res)
  dset_neg = f.create_dataset(f'NEG', data=res)
  dset_neg.attrs['shape'] = res.shape
  
  # close h5 file
  f.close() 