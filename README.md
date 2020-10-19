# SERP2021-BGSUB

A project to build a robust method for foreground-object/caribou image detection from camera trap footage, recorded in the Alaskan Wilderness. Our main method of achieving this would be utilizing [FgSegNet](https://github.com/lim-anggun/FgSegNet), a robust background subtraction model with a triple-CNN encoder-decoder architecture.

_**NOTE**_: _Large Files (including compressed archives and model files) are omitted from this repository. As a result, `img_raw_tar` and `FgSegNet_M` are empty directories in this repo._

## Folder Structure
 - `FgSegNet`: Files related to FgSegNet model
     - `FgSegNet/FgSegNet`: Modified [FgSegNet model source code](https://github.com/lim-anggun/FgSegNet)
     - `FgSegNet/FgSegNet_M`: Output model files
 - `csv_raw`: CSV files from TimeLapse dataset (contains extraneous information)
     - Filenames: `<folderName>_<#>.csv`
 - `csv_parsed`: CSV files that include file info that has at least one object of importance
     - Filenames: `<folderName>_<#>_parsed.csv`
 - `img_raw_tar`: Tar.gz files that contain all images from respective CSV files in `csv_parsed`
     - Filenames: `<folderName>_<#>_positive.tar.gz`
 - `img_DS`: Directories and zipped files containing raw and labeled images (labled locally)
     - Filenames: `<folderName><#>_DS(.zip)`
     
## Notable Files
 - `prudhoe_15_concat.ipynb`: Experimental Notebook to automate process of creating files in `csv_parsed` & `img_raw_tar` from `csv_raw` files
 - `parse.py`: Functional script for automating the above function (derived from Notebook)
 - `tarify.py`: Gets files from `csv_parsed` to make `tar.gz` files (derived from Notebook)
 - `mask.py`: Used locally to convert Labelme `.json`s into binary masks
 - `img_DS/prudhoe#_DS/move.py`: Moves files in DS directories into `scratch` for FgSegNet model
     - Reduces image dimensions by half (due to GPU memory limitations)

### Acknowledgements
I would like to thank Dr. Michael I Mandel from Brooklyn College CUNY as well as Dr. John Davis from Staten Island Technical High School for assisting, advising, and supervising me throughout this project.