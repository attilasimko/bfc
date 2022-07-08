# Implicit Training

A repository for the paper 'MRI bias field correction with an implicitly trained CNN.' accepted for MIDL 2022.

Link to paper: https://openreview.net/forum?id=LbHd47ij5s

<<<<<<< HEAD
The required packages to run the repository are collected in 'requirements.txt', and the Python version we used is 3.7.13
=======
Link to the Zenodo site with the trained model and an example MR scan: https://zenodo.org/record/3749526#.YpXdjL9ByV4

The required packages to run the repository are collected in 'requirements.txt'
>>>>>>> b84bd434113b754af9f5586a58c4afdc37b61d46

The code 'BFC_testing.py' shows the evaluation of the ImageNet-based model on an example image from BrainWeb. This model is the same as the one evaluated in the paper.

The code 'AWGN_training.py' shows how simple it is to implement implicit learning, instead of the case of bias field correction, here we present a training process for Gaussian Noise removal.
