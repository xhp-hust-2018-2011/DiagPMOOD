# FixOOD

# prepare the dataset
following openood to download the data, 
unzip the "openood.zip" in the current folder,
put all the openood data in the ./data folder,
download the "models.zip" and extract it in the current folder.


# for evaluation
## first extract the feature
run "sh extract_feature.sh"

## for augdelete you can
run "python augdelete.py"

## testing the OOD performance
run "python test_ood.py"

# for training
run "python train.py" with the arguments.
