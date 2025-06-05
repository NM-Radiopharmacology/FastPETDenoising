"""
script to establish training configuration
1) User chooses whether to use default training parameters or to set them up manually
2) User is asked to provide paths to folder with training images and folder with respective references
3) 3-fold cross-validation is performed for internal validation during training
OUTPUT: A configuration.json file is created --> 2_train.py will run from this file, therefore, this script must be
run beforehand
"""
from utilities import kfold_cross_validation, print2, create_configuration_file, get_dataset
import json


# Setting training configuration, including providing training pairs...
# Providing dataset: paths to folders with training images and references will be requested: images must be ordered
# and paired!!
configuration = create_configuration_file()
training_pairs = configuration["dataset"]["training_pairs"]

# 3-fold cross-validation for training
validation_folds = kfold_cross_validation(k=3, data=training_pairs)
configuration["dataset"]["validation_folds"] = validation_folds

# A configuration.json file is created with the training configuration (including training pairs)
configuration_json = json.dumps(configuration, indent=4)
open("configuration.json", "w").close()
with open("configuration.json", "w") as config_file:
    config_file.write(configuration_json)

print2("training configuration established")
