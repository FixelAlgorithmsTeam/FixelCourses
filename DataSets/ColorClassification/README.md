# Color Classification

Based on [Kaggle - Colour Classification](https://www.kaggle.com/datasets/trushraut18/colour-classification) by [Raut](https://www.kaggle.com/trushraut18).


## Processing

 - Converted `Green46.jpg` in `Train\Green` into regular `RGB` (Was `CMYK`).
 - Converted `Green900.jpg` in `Train\Green` into regular `RGB` (Was `CMYK`).
 - Cropped `Red721.jpg` in `Train\Red` As the original file was corrupted.
 - Converted `Blue10.jpg` in `Validation\Blue` into regular `RGB` (Was `Index`).
 - Converted `Blue2.jpg` in `Validation\Blue` into regular `RGB` (Was `Index`).
 - Converted `Blue4.jpg` in `Validation\Blue` into regular `RGB` (Was `Index`).
 - Converted `Blue48.jpg` in `Validation\Blue` into regular `RGB` (Was `Index`).
 - Converted `Blue54.jpg` in `Validation\Blue` into regular `RGB` (Was `Index`).
 - Converted `Blue55.jpg` in `Validation\Blue` into regular `RGB` (Was `Index`).
 - Run `ConvertDataSetMat.m` which resizes images and creates a single matrix.
 - Merged Train and Validation data sets.
