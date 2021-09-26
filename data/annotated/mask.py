from PIL import Image, ImageDraw
from pathlib import Path
from shutil import copy2
import os
import json

RAW_PTH = Path('DS/raw')
LABEL_PTH = Path('DS/label')

labels_gen = Path().rglob('*.json')
os.makedirs(RAW_PTH, exist_ok=True)
os.makedirs(LABEL_PTH, exist_ok=True)

for label in labels_gen:
    print(label)
    
    f = open(label)
    j = json.load(f)
    img = Image.new('L', (j["imageWidth"], j["imageHeight"]))
    draw = ImageDraw.Draw(img)

    for shape in j["shapes"]:
        poly = [(i[0], i[1]) for i in shape["points"]]
        draw.polygon(poly, fill=255)

    raw_pth = (RAW_PTH / label).as_posix()[:-5] + '.JPG'
    label_pth = (LABEL_PTH / label).as_posix()[:-5] + '.PNG'
    
    os.makedirs((RAW_PTH / os.path.dirname(label)), exist_ok=True)
    os.makedirs((LABEL_PTH / os.path.dirname(label)), exist_ok=True)
    
    img.save(label_pth)
    copy2(f'{str(label)[:-5]}.JPG', raw_pth)

    f.close()
