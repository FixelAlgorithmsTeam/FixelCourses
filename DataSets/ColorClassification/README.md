# Color Classification

[![Visitors](https://img.shields.io/badge/Fixel-Algorithms-blue)](https://fixelalgorithms.gitlab.io)
[![Visitors](https://api.visitorbadge.io/api/combined?path=https%3A%2F%2Fgithub.com%2FRoyiAvital%2FStackExchangeCodes&labelColor=%23f47373&countColor=%23555555&style=plastic)](https://github.com/FixelAlgorithmsTeam/FixelCourses) <!-- https://www.visitorbadge.io -->

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
