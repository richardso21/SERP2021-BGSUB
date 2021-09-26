from PIL import Image
from pathlib import Path
import os

dst = 'DS_half'

os.makedirs(dst, exist_ok=True)
os.makedirs(os.path.join(dst, 'label'), exist_ok=True)
os.makedirs(os.path.join(dst, 'raw'), exist_ok=True)

fns = Path('DS/').rglob('*.*G')
for i in fns:

    img = Image.open(i)
    img = img.resize((img.width // 2, img.height // 2))

    new_fn = str(i)[3:].replace('/','_')
    typ = 'label' if (new_fn[:5] == 'label') else 'raw'

    img.save(os.path.join(dst, typ, new_fn))
    print(i)
