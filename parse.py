import pandas as pd
from pathlib import Path
import os

# get all csv files in `csv_raw`
csv_raw = Path('csv_raw').glob('*.csv')

for csv in csv_raw:
  print(f'Parsing {csv}')

  # find path on dataset w/ csv name
  basename = csv.name.strip('.csv')
  location, number = basename.split('_')

  # check if parsed file already exists
  fname = f'csv_parsed/{location}_{number}_parsed.csv'
  if os.path.isfile(fname):
    print('already exists. skipping...')
    continue

  # read csv file
  F = pd.read_csv(csv)
  tank_pth = os.path.join('/tank/data/nna/cameraTrap', location, number)

  # add the furthest subdirectories if they contain files
  img_dirs = []
  for dirpath, dirnames, filenames in os.walk(tank_pth):
    if not dirnames and filenames:
      img_dirs.append(dirpath)
  img_dirs.sort()

  # associate referred rel dirs on csv with dirs on local dataset
  rel_paths = {}
  for u, v in zip(F.RelativePath.unique(), img_dirs):
    rel_paths[u] = v

  # categories to check for object presence
  categories = ["Caribou", "Bear", "Fox", "Wolf", "Muskox",
              "Misc_Land_Animal", "Human", "Other", "Waterfowl",
              "Upland_Game_Bird", "Songbird", "Sea_Shore_Bird"]

  # if categories is not zero & exists, append to list, convert list to DataFrame
  fs = []
  for i, row in F.iterrows():
    for c in categories:
        try:
          # check if there are objects of importance (count > 0)
          if row[c] > 0:
            fs.append([row["File"],
                        rel_paths[row["RelativePath"]].replace(tank_pth+'/', ''),
                        os.path.join(rel_paths[row["RelativePath"]],row["File"]),
                        c])
        # pass if the category doesn't exist
        except KeyError:
            pass

  # turn into DataFrame, and save to `csv_parsed` directory
  fs_df = pd.DataFrame(fs, columns=["fileName", "RelFilePath", "AbsFilePath", "Type"])
  fs_df.to_csv(fname)
  print(f'Parsed {csv}')
