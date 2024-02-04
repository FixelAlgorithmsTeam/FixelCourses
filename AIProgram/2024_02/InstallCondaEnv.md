# Install the Course `conda` Environment

This guide is mainly Windows users though the main steps can be replicated in `macOS` and `Linux`.

## Auxiliary Steps

1. Update Windows  
   Make sure your Windows version is updated using `Windows Update`.   
   For Windows 10 it should be `Windows 10 Version 22H2`.
2. Install Windows Terminal  
   It suggested to install [Windows Terminal](https://github.com/microsoft/terminal).  
   If can be done, the best way is to install it using the Windows Store: [Windows Store - Windows Terminal](https://apps.microsoft.com/detail/9N0DX20HK701).  
   It can be installed by downloading the latest official release from GitHub and using `Add-AppPackage` on `PowerShell Terminal`.  
   Once installed, set it to be the default console application on Windows.
3. Install VS Code  
   You should install VS Code form the [VS Code Download Page](https://code.visualstudio.com/download).  
   Choose the `User Installer`. Make sure to chose to add `Code` to path one options are showed.


# Install the Conda Environment

This is a basic instruction how to install the course environment under Anaconda (Or any `conda` managed distribution).  
It is assumed that the Windows OS is used.

1. Install the [MiniForge](https://conda-forge.org/miniforge/).  
   It is better to download the `Mamba` based versions: `Mambaforge-xxxx`. 
2. Download the environment specification file from: [The `conda` environment file](https://github.com/FixelAlgorithmsTeam/FixelCourses/blob/master/AIProgram/2024_02/EnvTechnionAiProg.yml).  
   Save it as `EnvTechnionAiProg.yml`.
3. Open `Conda Prompt (conda)`. You should see something like `(base) ...` on the terminal:
![](https://i.imgur.com/AGDV0WF.png)
4. Navigate to the folder where `EnvTechnionAiProg.yml` is located.
5. Run the command: `conda create --file EnvTechnionAiProg.yml`. It should take a while.
6. Once it is finished, run `conda activate TechnionAiProg`. You should see `(base)` changes into `(TechnionAiProg)`.
7. Open VS Code (Run `code` on command line).
8. Open the folder of the notebooks of the course.
9. Open a notebook and make sure you set the Python Kernel to `TechnionAiProg` (It should say Python `3.11.7` or `3.11.8`).