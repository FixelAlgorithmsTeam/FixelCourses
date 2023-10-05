# Install the Conda Environment

This is a basic instruction how to install the course environment under Anaconda (Or any `conda` managed distribution).  
It is assumed that the Windows OS is used.

1. Install the [Anaconda distribution](https://www.anaconda.com/products/distribution).  
   It should install VS Code. If not, you may install it manually (Remember to install the Microsoft Python Extension). 
2. Download the environment specification file from: [The `conda` environment file](https://raw.githubusercontent.com/FixelAlgorithmsTeam/FixelCourses/master/ImageProcessingPython/CourseEnv.yml).  
   Save it as `CourseEnv.yml`.
3. Open `Anaconda Prompt (conda)`. You should see something like `(based) ...` on the terminal:
![](https://i.imgur.com/AGDV0WF.png)
4. Navigate to the folder where `CourseEnv.yml` is located.
5. Run the command: `conda env create -f CourseEnv.yml`. It should take a while.
6. Once it is finished, run `conda activate ImageProcessingPython`. You should see `(base)` changes into `(ImageProcessingPython)`.
7. Open VS Code.
8. Open the folder of the notebooks of the course.
9. Open a notebook and make sure you set the Python Kernel to `ImageProcessingPython`.

**Remark 001**: Advanced users may use `miniconda` to achieve the same result.  
**Remark 002**: Advanced users may use `micromamba` to achieve the same result.  