# DiagPMOOD

# intsall the python environment
run "conda env create -f environment.yml" and "pip install dotmap tqdm".

# prepare the dataset
following [OpenoodV1.5](https://github.com/Jingkang50/OpenOOD) to download the data, 

put all the openood data in the ./data folder, 

it should lool like as "./data/benchmark_imglist ./data/images_classic ./data/images_largescale",

also need the ImageNet-1k training data to be put in the folder "./data/images_largescale/imagenet_1k/train";

unzip the "openood.zip" in the current folder as "./openood",

download the "[models.zip](https://drive.google.com/file/d/1pFWlH3TVEcLnyjWyos_KKv51bMDcQGUB/view?usp=sharing)" and unzip it in the current folder as "./models".


# for evaluation
## first extract the feature
run "sh extract_feature.sh"

## for augdelete you can
run "python augdelete.py"

## testing the OOD performance
run "python test_ood.py"

# for training
run "python train.py" with the arguments.
