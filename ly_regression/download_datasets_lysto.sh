/*
Download training and test dataset from zenodo.
Datasets refer to lysto grand challenge:
https://lysto.grand-challenge.org/LYSTO/

Datasets consist of IHC images tiles from 40x magnification level
training: 2E4 images
test: 1.2E4 images
*/
wget -O training.h5 https://zenodo.org/record/3513571/files/training.h5?download=1
wget -O test.h5 https://zenodo.org/record/3513571/files/test.h5?download=1