import pandas as pd
from pathlib import Path
import os
import tarfile

# get all csv files in `csv_parsed`
csv_parsed = Path('csv_parsed').glob('*.csv')

for csv in csv_parsed:
  print(f'Taring {csv}')

  # define tarfile filename
  tf_fn = str(csv).replace('parsed.csv', 'positive.tar.gz'
                   ).replace('csv_parsed','img_raw_tar')
  if os.path.isfile(tf_fn):
    print('already exists. skipping...')
    continue

  # create new tarfile
  tf = tarfile.open(tf_fn, mode='x:gz')

  # read parsed csv file
  F = pd.read_csv(csv)

  e = 0
  for i, row in F.iterrows():
    try:
      tf.add(row["AbsFilePath"],
             arcname=os.path.join(row["RelFilePath"],row["fileName"]))
    except:
      e += 1
  print(f'{e} error(s) encountered when taring {csv}. [Probably missing files]')
  tf.close()
