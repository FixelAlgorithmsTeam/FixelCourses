---
    # Meta Data
    Title: Reichman University (RUNI) - UnSupervised Learning Methods (3655) - Syllabus
    Summary: The syllabus for the course UnSupervised Learning Methods (3655)
    Author:
        - Royi Avital (Royi.Avital@post.runi.ac.il)

    # Settings
    settings:
        enable_uml: true
        image_path_conversion: base64
        markdown_extensions:
          - markdown.extensions.footnotes
          - markdown.extensions.fenced_code
          - markdown.extensions.tables
          - markdown.extensions.admonition
          - markdown.extensions.md_in_html
          - markdown.extensions.toc:
                title: Table of Contents
                toc_depth: 2-3
          - pymdownx.arithmatex:
                generic: true
                smart_dollar: false
        allow_css_overrides: true
---


# Syllabus: Reichman University (RUNI) - UnSupervised Learning Methods (3655)

[TOC]


## Description

The course deals with the methods of Unsupervised Learning and its applications in the field of data science.

The course will cover the following subjects:

 * Essentials
    - Linear Algebra.
    - Probability (Video).
    - Optimization (Convex).
 * Clustering
    - Parametric Methods: K-Means, K-Medoids, Gaussian Mixture Models.
    - Non Parametric Methods: Hierarchical Clustering, DBSCAN, HDBSCAN, Spectral Clustering.
 * Dimensionality Reduction & Manifold Learning.
    - UnSupervised Methods: PCA, Kernel PCA, MDS, IsoMap, Laplacian Eigen Maps, t-SNE.
    - Supervised Methods[^001]: LDA, QDA.
 * Anomaly Detection
    - Statistical Tests.
    - Local Outlier Factor.
    - Isolation Forest.

The course will have a brief introduction to _Deep Learning_ in the context of Auto Encoders.  
It won't cover the whole field, but only a sub set needed for the context of the course.


## Grading

 * The grading of the course will be based on 4 exercises with maximum grade per exercise being `100`.  
 * Work on the exercises is in group. The maximum size of the group will be set according to amount of students in class.
 * The exercises might include both theoretic question and programming tasks.

  [^001]: If time permits.