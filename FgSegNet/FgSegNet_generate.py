import numpy as np
import os
from os.path import join
from pathlib import Path
from keras.preprocessing import image as kImage
from skimage.transform import pyramid_gaussian
from keras.models import load_model
from my_upsampling_2d import MyUpSampling2D

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Define paths to files and output directory
PTH = '../data/classifier'
OPTH = join(PTH, 'FgSegNet_O')
SCENE = 'BGSUB'

# scales images to three proportions (following FgSegNet methods)
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

Path(OPTH).mkdir(exist_ok=True)
Path(join(OPTH, SCENE)).mkdir(exist_ok=True)

# load respective model
mdl_path = join(PTH, 'models', f'mdl_{SCENE}.h5')

model = load_model(mdl_path, custom_objects={'MyUpSampling2D': MyUpSampling2D}, compile=False)

# create positive + negative image list
input_pths_pos = [pth for pth in Path(join(PTH, 'pos')).rglob('*.JPG')]
input_pths_neg = [pth for pth in Path(join(PTH, 'neg')).rglob('*.JPG')]

# separate them into chunks (prevent memory overload)
input_splts_pos = [input_pths_pos[x:x+32] for x in range(0, len(input_pths_pos), 32)]
input_splts_neg = [input_pths_neg[x:x+32] for x in range(0, len(input_pths_neg), 32)]

# for each list of splits
for neg_bool, input_splts in enumerate([input_splts_pos, input_splts_neg]):
    # for each chunk
    T = 'neg' if neg_bool else 'pos'
    Path(join(OPTH, SCENE, T)).mkdir(exist_ok=True)
    for input_splt in input_splts:
        # scale the images w/ `scaleImg()`
        data = scaleImg(input_splt)

        # feed them into model & reshape
        probs = model.predict(data, batch_size=1, verbose=1)
        probs = probs.reshape([probs.shape[0], probs.shape[1], probs.shape[2]])
        
        # save each probability mask as .npy file w/ respective name
        for i, input_fn in enumerate(input_splt):
            with open(join(OPTH, SCENE, T, f'{input_fn.stem}.npy'), 'wb') as F:
                np.save(F, probs[i])
                print(F.name)
