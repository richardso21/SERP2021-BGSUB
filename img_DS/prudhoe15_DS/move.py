from shutil import copy2
from PIL import Image
from pathlib import Path
import os

dst = '/home/richardso21/scratch/20-21_BGSUB/prudhoe15_DS_dst_half'

os.makedirs(dst, exist_ok=True)
os.makedirs(os.path.join(dst, 'label'), exist_ok=True)
os.makedirs(os.path.join(dst, 'raw'), exist_ok=True)

fns = Path().rglob('*.*G')
for i in fns:

    img = Image.open(i)
    img = img.resize((img.width // 2, img.height // 2))

    new_fn = str(i).replace('/','_')
    typ = 'label' if (new_fn[-3:] == 'PNG') else 'raw'

    img.save(os.path.join(dst, typ, new_fn))
    # copy2(i, os.path.join(dst, typ, new_fn))
    print(i)
    # break