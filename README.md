<h1 align="center">  8P361 - Group 13  </h1>

This github repository contains the code used to perform our research. We compared 2 different capsule network routings and a simple CNN model.
The architecture of the two capsule networks can be found in the models folder. They can both be trained in the efficient_capsnet_train file by specifying the model name as either "self_attention" or "dynamic_routing". The hyperparameters can be changed in the config.json file. The code of the capsule layers themselves can be found in the utils folder in layers and layers_hinton for the self attention and dynamic routing respectively.
The CNN model was implemented in its entirety in the CNN_model.py file. All parameter sets can be trained back-to-back in this file.

To allow for the capsule networks to take in the images, they needed to converted to jpeg format. A file for this task is provided as well.


The original code was adapted from the paper "Efficient-CapsNet: Capsule Network with Self-Attention Routing", which has been linked below. The corresponding github can be found [here](https://github.com/EscVM/Efficient-CapsNet).

[![arXiv](http://img.shields.io/badge/arXiv-2001.09136-B31B1B.svg)](https://arxiv.org/abs/2101.12491)
