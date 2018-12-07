# irasl2018
Code for the paper ["Investigating the effect of residual and highway connections in speech enhancement models"](https://openreview.net/forum?id=rkzeXBDos7)

For generating the dataset: 
    - Download the [IEEE dataset](https://www.crcpress.com/downloads/K14513/K14513_CD_Files.zip) and save it to data/IEEE_dataset. Copy sentence lists 01 to 67 to IEEE_dataset/train and 68 to 72 to IEEE_dataset/test. 
    - Download the [DEMAND](https://zenodo.org/record/1227121) and save it to data/DEMAND.
    - Run the script gen_noisy_dataset.py.
    - Use the script create_desc_json_noisy.py to create a JSON file that defines the dataset for the training script.

For training the models, use main.py. For testing the model and generating visualizations, use test_model_vis.py. Both scripts have a list of parameters that define the model type, number of blocks, GRU layers per block, and other parameters.
