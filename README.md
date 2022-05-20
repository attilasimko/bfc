# Implicit Training

A repository for the paper 'MRI bias field correction with an implicitly trained CNN.' accepted for MIDL 2022.

The required packages to run the repository are collected in 'requirements.txt'

The code 'BFC_testing.py' shows the evaluation of the ImageNet-based model on an example image from BrainWeb. This model is the same as the one evaluated in the paper.

The code 'AWGN_training.py' shows how simple it is to implement implicit learning, instead of the case of bias field correction, here we present a training process for Gaussian Noise removal.
