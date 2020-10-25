import numpy as np
import os
from os.path import join
from pathlib import Path
from keras.preprocessing import image as kImage
from skimage.transform import pyramid_gaussian
from keras.models import load_model
from my_upsampling_2d import MyUpSampling2D

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def scaleImg(input_path):
    X = []
    for p in input_path:
        x = kImage.load_img(p)
        x = kImage.img_to_array(x)
        X.append(x)
    X = np.asarray(X)

    s1 = X
    del X
    s2 = []
    s3 = []
    for x in s1:
        pyramid = tuple(pyramid_gaussian(x/255., max_layer=2, downscale=2))
        s2.append(pyramid[1]*255.)
        s3.append(pyramid[2]*255.)
    
    s2 = np.asarray(s2)
    s3 = np.asarray(s3)
    
    return [s1, s2, s3]

PTH = '/scratch/richardso21/20-21_BGSUB'
OPTH = join(PTH, 'FgSegNet_O')
SITES = ['prudhoe_12', 'prudhoe_15', 'prudhoe_22']
TSET = 200

Path(OPTH).mkdir(exist_ok=True)

for site in SITES:
    print(f'SITE: {site}')
    mdl_path = join(PTH, f'FgSegNet/FgSegNet_M/Prudhoe/models{TSET}', f'mdl_{site.replace("_", "")}.h5')
    model = load_model(mdl_path, custom_objects={'MyUpSampling2D': MyUpSampling2D}, compile=False)
    
    input_pths_pos = [pth for pth in Path(join(PTH, 'FgSegNet_Test', site, 'pos')).rglob('*.JPG')]
    input_pths_neg = [pth for pth in Path(join(PTH, 'FgSegNet_Test', site, 'neg')).rglob('*.JPG')]

    input_splts_pos = [input_pths_pos[x:x+32] for x in range(0, len(input_pths_pos), 32)]
    input_splts_neg = [input_pths_neg[x:x+32] for x in range(0, len(input_pths_neg), 32)]

    for i, input_splt in enumerate(input_splts_pos):
        data = scaleImg(input_splt)

        probs = model.predict(data, batch_size=1, verbose=1)
        probs = probs.reshape([probs.shape[0], probs.shape[1], probs.shape[2]])
        
        with open(join(OPTH, f'{site}_pos_{i+1}.npy'), 'wb') as F:
            np.save(F, probs)
            print(F.name)

    for i, input_splt in enumerate(input_splts_neg):
        data = scaleImg(input_splt)

        probs = model.predict(data, batch_size=1, verbose=1)
        probs = probs.reshape([probs.shape[0], probs.shape[1], probs.shape[2]])
        
        with open(join(OPTH, f'{site}_neg_{i+1}.npy'), 'wb') as F:
            np.save(F, probs)
            print(F.name)