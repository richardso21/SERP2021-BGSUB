import os
from os.path import join, isfile
from pathlib import Path
# from shutil import copy2
import pandas as pd
import random
from tqdm import tqdm
from PIL import Image

SITES = ['prudhoe_12', 'prudhoe_15','prudhoe_22']
ENDPTH = '/scratch/richardso21/20-21_BGSUB/FgSegNet_Test_wNeg'

Path(ENDPTH).mkdir(exist_ok=True)

def img_half(pth, save_pth):
  img = Image.open(pth)
  img = img.resize((img.width // 2, img.height // 2))

  img.save(save_pth)

for site in SITES:

  print(f"CURRENT SITE: {site}")
  # make dirs @ endpath
  end_dir = join(ENDPTH, site)
  pos_dir = join(end_dir, 'pos')
  neg_dir = join(end_dir, 'neg')
  Path(end_dir).mkdir(exist_ok=True)
  Path(join(pos_dir)).mkdir(exist_ok=True)
  Path(join(neg_dir)).mkdir(exist_ok=True)
  # open csv file containing all positives of the site
  df = pd.read_csv(f'csv_parsed/{site}_parsed.csv')
  # get names of all positive images 
  positives = [join(x['RelFilePath'], x['fileName']) for n, x in df.iterrows()]
  # get # of all positive images 
  pos_len = len(positives)
  # get path to relative DS directory (@ `img_ds/`)  
  img_DS_pth = join('img_DS', f'{site.replace("_","")}_DS', 'raw')
  # get path to relative DS directory (@ `img_ds/`)
  tank_pth = f'/tank/data/nna/cameraTrap/prudhoe/{site.strip("prudhoe_")}'


  print(f"{site} | Positive files")
  c = 0
  e = 0
  for positive in tqdm(positives):
    # only files not found in DS (not manually labeled)
    if isfile(join(img_DS_pth, positive)):
      continue
    # make new filename and put in `pos_dir`
    new_fn = positive.replace('/','_')
    try:
      # copy2(join(tank_pth, positive), join(pos_dir, new_fn))
      img_half(join(tank_pth, positive), join(pos_dir, new_fn))
      c += 1
    except:
      e += 1
  print(f'Positive unlabeled w/o error: {c}')
  print(f'Errors (missing files): {e}')


  print(f"{site} | Negative files")
  # find all images in the dataset
  tank_all = [x for x in Path(tank_pth).rglob("*.JPG")]
  # open csv file containing negatives already used for training
  df_neg = pd.read_csv(f'csv_negative/{site}_negative.csv')
  i = 0
  pbar = tqdm(total=(c * 2))
  # continue for 2x amount of positives
  while i < (c * 2):
    # make random choice in dataset
    negative = str(random.choice(tank_all))
    # neglect if in positives list
    if negative in df['AbsFilePath']:
      continue
    # neglect if in negatives-already-used list
    if negative in df_neg['AbsFilePath']:
      continue
    # make new filename and put in `neg_dir`
    new_fn = negative.replace(tank_pth, '').replace('/','_').strip('_')
    # copy2(negative, join(neg_dir, new_fn))
    img_half(negative, join(neg_dir, new_fn))
    # + 1
    pbar.update(1)
    i += 1
  pbar.close()
  # break