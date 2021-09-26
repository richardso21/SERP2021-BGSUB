# SERP2021-BGSUB

A project to build a robust method for foreground-object/caribou image detection from camera trap footage, recorded in the Alaskan Wilderness. Our main method of achieving this would be utilizing [FgSegNet](https://github.com/lim-anggun/FgSegNet), a robust background subtraction model with a triple-CNN encoder-decoder architecture.

## Workflow (Usage Instructions)

### Prerequisites

> Ensure your computing system has at least **10 GB of GPU Memory/VRAM** so FgSegNet won't crash! _You may tinker with FgSegNet's batch size if you have limited resources._

1. Install the latest version of [Anaconda](https://www.anaconda.com/products/individual).

    - _This will be needed to quickly manage the required dependencies the project's code will need._

    &nbsp;

2. Install [labelme](https://github.com/wkentaro/labelme) on a machine **where you can and will annotate sample images**.

    - _labelme will be used to manually generate training data._

### Setting up the environment

1. Clone this project onto your machine (using git or manually as a `.zip`):

```
git clone https://github.com/richardso21/SERP2021-BGSUB.git
```

2. Utilize your anaconda install to create a new Python "environment" for the project's code to run:

```
cd SERP2021-BGSUB
conda env create --file environment.yml
conda activate bgsub
```

3. Replace the existing `pyramids.py` of the scikit-image library in the environment with the patched file in this repository. This neat command will do this for you:

```
cp pyramids.py $(conda info | grep "active env location" | cut -d ' ' -f 9)/lib/python3.6/site-packages/skimage/transform/pyramids.py
```

-   _Run this command with `sudo` if you have a permission error._

### Labeling/Preparing Image Data

> NOTE: Your raw image data **should** be in the `.JPG` format for the project code to code properly.

1. Use [labelme](https://github.com/wkentaro/labelme) to draw polygonal annotations of foreground objects for a pool of foreground-positive image samples.

    - **Your images should be contained in the directory named `data/annotated` at the root of this repository.**
    - The recommended number of images to annoatate is ≈200 images, although you can annotate less/more according to your data or needs.

    &nbsp;

2. Run `mask.py` (`python mask.py`) in the `data/annotated` directory which contains the images.

3. _(Recommended) Additionally, run `resize.py` to shrink the size of the raw/label image data if they are very large, since that can also lead to FgSegNet crashing when attempting to train._

### Training the Annotator (FgSegNet)

1. Check and tune the hyperparameters(variables) in the `FgSegNet.py` script.

    - You can change the `scene` name to differentiate between models for different image sets.
    - Change `num_frames` to match the number of annotated image samples.
    - Alter `batch_size` according to your computing resources (smaller batch size requies less resources).

    &nbsp;

2. Run `FgSegNet.py` in the `FgSegNet` directory. It will automatically use your annotated image data to train a new model.
    - FgSegNet will generate a lot of terminal output, namely for debugging and process tracking purposes.
    - If FgSegNet sucessfully trains, **a new directory `models` will contain the model file in the format `mdl_<scene_name>.h5`.**

### Training the Foreground Presence Predictor

**TODO**

---

### Acknowledgements

I would like to thank Dr. Michael I Mandel from Brooklyn College CUNY as well as Dr. John Davis from Staten Island Technical High School for assisting, advising, and supervising me throughout this project.
