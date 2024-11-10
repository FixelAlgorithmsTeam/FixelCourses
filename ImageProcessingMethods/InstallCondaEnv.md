# Install the Conda Environment

This is a basic instruction how to install the course environment using [Mini Conda Forge](https://github.com/conda-forge/miniforge) (Or any `conda` managed distribution).  
It is assumed that the Windows OS is used.

1. Install the [Mini Conda Forge Distribution](https://github.com/conda-forge/miniforge/releases/latest).  
   Choose the latest version based on Mamba. For Windows it should be in the template: ` Mambaforge-<versionString>-Windows-x86_64.exe `.   
   It should create an entry in the menu called `Miniforge Prompt`.
2. Install [VS Code](https://code.visualstudio.com).  
   After installation install the following extensions by MS (Microsoft): `Python`, `Jupyter`, `Pylance`.
3. Download the environment specification file from: [The `conda` environment file](https://raw.githubusercontent.com/FixelAlgorithmsTeam/FixelCourses/master/ImageProcessingMethods/CourseEnv.yml).  
   Save it as `CourseEnv.yml`. 
4. Open `Miniforge Prompt`. You should see something like `(based) ...` on the terminal:
![](https://i.imgur.com/AGDV0WF.png)
5. Navigate to the folder where `CourseEnv.yml` is located.
6. Run the command: `conda env create -f CourseEnv.yml`. It should take a while.
7. Once it is finished, run `conda activate ImageProcessingMethods`. You should see `(base)` changes into `(ImageProcessingPython)`.
8. Open VS Code (Run `code` in the command line where the environment is activated).
9. Open the folder of the notebooks of the course.
10. Open a notebook and make sure you set the Python Kernel to `ImageProcessingMethods`.

**Remark 001**: Advanced users may use `CourseEnvIntel.yml` / `CourseEnvMacOS.yml` to have optimized `BLAS` / `LAPACK` libraries.  
**Remark 002**: Advanced users may use `micromamba` to achieve the same result.  