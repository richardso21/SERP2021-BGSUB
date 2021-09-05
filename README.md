# SERP2021-BGSUB

A project to build a robust method for foreground-object/caribou image detection from camera trap footage, recorded in the Alaskan Wilderness. Our main method of achieving this would be utilizing [FgSegNet](https://github.com/lim-anggun/FgSegNet), a robust background subtraction model with a triple-CNN encoder-decoder architecture.

_**NOTE**_: _Large Files (including compressed archives and model files) are omitted from this repository. As a result, `img_raw_tar` and `FgSegNet_M` are empty directories in this repo._

## Workflow (Usage Instructions)
1. Clone this repository into a system that will be used for ML model training. _(Make sure that your system is powerful enough to handel neural network training; FgSegNet's architecture is especially large!)_
2. Make sure you have the latest version of [Anaconda](https://www.anaconda.com/products/individual)/Miniconda installed onto your system. On this directory, run the following commands line-by-line:
```
conda env create --file environment.yaml
conda activate bgsub
```
_This will create a conda environment called `bgsub` with the proper Python and library/dependency versions._  

3. Replace (patch) the existing `pyramids.py` of the scikit-image library with the one in `FgSegNet/FgSegNet/skimage_pyramids.py`. `pyramids.py` should be located in `<path_to_conda>/conda/envs/bgsub/lib/python3.6/site-packages/skimage/transform/pyramids.py`.
4. Utilize the open-source software [labelme](https://github.com/wkentaro/labelme) to draw polygonal annotations of foreground objects for ≈200 foreground-positive images. The label names of the annotations do not matter. _This procedure is necessary as FgSegNet requires training data to be effective in foreground segmentation._
5. Labelme generates a `.json` file for each labeled image. Run `mask.py` on the directory which your images/JSON files are stored; it should generate a new folder `DS` within the directory. 
6. TODO


## Folder Structure
 - `FgSegNet`: Files related to FgSegNet model
     - `FgSegNet/FgSegNet`: Modified [FgSegNet model source code](https://github.com/lim-anggun/FgSegNet)
     - `mask_classifier`: Classifier for FgSegNet's outputs
        - `old` : graphs and data derived from FgSegNet without negative label training (old method)
        - `neg` : graphs derived from FgSegNet w/ negative label training
 - `csv_raw`: CSV files from TimeLapse dataset (contains extraneous information)
     - Filenames: `<folderName>_<#>.csv`
 - `csv_parsed`: CSV files that include file info which has at least one object of importance
     - Filenames: `<folderName>_<#>_parsed.csv`
 - `img_raw_tar`: Tar.gz files that contain all images from respective CSV files in `csv_parsed`
     - Filenames: `<folderName>_<#>_positive.tar.gz`
 - `img_DS`: Directories and zipped files containing raw and labeled images (labled locally)
     - Filenames: `<folderName><#>_DS(.zip)`
 - `csv_negative`: CSV files that include file info for all negatively-labeled imgs put for FgSegNet training
     - Filenames: `<folderName>_<#>_negative.csv`
     
## Notable Files
 - `FgSegNet/FgSegNet/FgSegNet_generate_prudhoe.py` : Script written by author to produce FgSegNet masks out of trained scene-specific models
 - `FgSegNet/mask_classifier/presence_thr(_neg).ipynb` : Experimental Jupyter nb to investigate generated data
 - `FgSegNet/mask_classifier/presence_thr_class.ipynb` : Improved (decluttered) notebook of the former using abstracted data classes
 - `prudhoe_15_concat.ipynb`: Experimental Notebook to automate process of creating files in `csv_parsed` & `img_raw_tar` from `csv_raw` files
     - `parse.py`: Functional script for automating the above function (derived from Notebook)
     - `tarify.py`: Gets files from `csv_parsed` to make `tar.gz` files (derived from Notebook)
 - `mask.py`: Used locally to convert Labelme `.json`s into binary masks
 - `img_DS/prudhoe#_DS/move.py`: Moves files in DS directories into `scratch` for FgSegNet model
     - Reduces image dimensions by half (due to GPU memory limitations)
 - `trainNegData.py`: Selects non-foreground images, 2x the amount of masked imgs in train set (400 negatives for 200 masked imgs already in set)
     - Also reduces image dimensions, & creates blank ground-truth masks for model to train on
     - Saves these selected images as CSVs in `csv_negative`
 - `testData.py`: Selects non-manually-masked images labeled to have foreground objects, and randomly picks 2x the amount of non-foreground images
     - Results in 1/3 positive and 2/3 negative images, which better simulates the proportions of images in dataset than 1/2 split.
     - Makes sure not to select images already used for training via `csv_negative` reference CSVs
 - `npy_to_h5.py|ipynb`: Script|notebook to convert npy outputs from `FgSegNet_generate_prudhoe.py` to (faster) h5 datasets

### Acknowledgements
I would like to thank Dr. Michael I Mandel from Brooklyn College CUNY as well as Dr. John Davis from Staten Island Technical High School for assisting, advising, and supervising me throughout this project.
