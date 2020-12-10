import os
from os.path import join, isfile
from pathlib import Path
# from shutil import copy2
import pandas as pd
import random
from tqdm import tqdm
from PIL import Image

SITES = ['prudhoe_12', 'prudhoe_15', 'prudhoe_22']
ENDPTH = '/scratch/richardso21/20-21_BGSUB/FgSegNet_Train'
CSVPTH = 'csv_negative'

# amount of positively-labeled items currently in training data
# AMNT = 200

def img_half(pth, save_pth):
  img = Image.open(pth)
  img = img.resize((img.width // 2, img.height // 2))

  img.save(save_pth)

def empty_img(pth, save_pth):
  # saves an empty image w/ same dimensions as inputted image
  ref = Image.open(pth)
  img = Image.new('RGB', (ref.width // 2, ref.height // 2))

  img.save(save_pth)


for site in SITES:
  print(f"CURRENT SITE: {site}")
  # keep array of filenames for tracking negative imgs used
  fs = []
  # define target directory
  target_dir = join(ENDPTH, f"{site.replace('_','')}_DS_dst_half_neg")
  # open csv containing all positive-labeled items in site
  pos_df = pd.read_csv(f'csv_parsed/{site}_parsed.csv')
  # get path to relative DS directory (@ `img_ds/`)
  tank_pth = f'/tank/data/nna/cameraTrap/prudhoe/{site.strip("prudhoe_")}'

  # find all images in the dataset
  tank_all = [x for x in Path(tank_pth).rglob("*.JPG")]

  i = 0
  amnt = len(os.listdir(join(target_dir, 'raw')))
  pbar = tqdm(total=(amnt * 2))
  # continue for 2x `AMNT`
  while i < (amnt * 2):
    # make random choice in tank
    negative = str(random.choice(tank_all))
    # neglect if in positives list
    if negative in pos_df['AbsFilePath']:
      continue
    
    # make new filename and put in `raw` 
    new_fn = negative.strip(tank_pth).replace('/','_')
    img_half(negative, join(target_dir, "raw", f"raw_{new_fn}"))
    # generate corresponding blank image (negative-labeled) at `label_rand...` folder
    # also, switch format to PNG format!!!
    empty_img(negative, join(target_dir, f"label", f"label_{new_fn.replace('JPG', 'PNG')}"))
    # append to `fs` array
    fs.append([new_fn, negative])
    i += 1
    pbar.update(1)
  pbar.close()
  # turn into DataFrame, and save to `csv_negative` directory
  fs_df = pd.DataFrame(fs, columns=["fileName", "AbsFilePath"])
  fs_df.to_csv(join(CSVPTH, f'{site}_negative.csv'))