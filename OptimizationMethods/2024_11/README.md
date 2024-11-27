# Fixel Courses - Optimization Methods

[![](./FixelAlgorithmsLogo.png)](https://fixelalgorithms.gitlab.io)

The [_Optimization Methods_](https://fixelalgorithms.gitlab.io/courses/optimizationmethods) course by [Fixel Algorithms](https://fixelalgorithms.gitlab.io).

[![Visitors](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FRoyiAvital%2FStackExchangeCodes&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=Visitors+%28Daily+%2F+Total%29&edge_flat=false)](https://github.com/FixelAlgorithmsTeam/FixelCourses)

This course class is using Python.

## Optimization Methods `2024_11`

This folder is dedicated to the course [`Optimization Methods`](https://fixelalgorithms.gitlab.io/courses/optimizationmethods) which took place on December 2023.  
The course was conducted in Python.

## Install the Course `conda` Environment

This guide is mainly Windows users though the main steps can be replicated in `macOS` and `Linux`.

### Auxiliary Steps

1. Update Windows  
   Make sure your Windows version is updated using `Windows Update`.   
   For Windows 10 it should be `Windows 10 Version 22H2` or more.
2. Install Windows Terminal  
   It suggested to install [Windows Terminal](https://github.com/microsoft/terminal).  
   If can be done, the best way is to install it using the Windows Store: [Windows Store - Windows Terminal](https://apps.microsoft.com/detail/9N0DX20HK701).  
   It can be installed by downloading the latest official release from GitHub and using `Add-AppPackage` on `PowerShell Terminal`.  
   Once installed, set it to be the default console application on Windows.
3. Install VS Code  
   You should install VS Code form the [VS Code Download Page](https://code.visualstudio.com/download).  
   Choose the `User Installer`. Make sure to chose to add `Code` to path one options are showed.
4. Enable Long Path Support  
   See [Microsoft - Maximum Path Length Limitation](https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation).  
   The operation requires elevated privileges.
   
### Install the Conda Environment

This is a basic instruction how to install the course environment under Anaconda (Or any `conda` managed distribution).  
It is assumed that the Windows OS is used.

1. Install the [MiniForge](https://conda-forge.org/miniforge).  
   It is better to download the `Mamba` based versions: `Mambaforge-xxxx`. 
2. Download the environment specification file from: [The `conda` environment file](https://github.com/FixelAlgorithmsTeam/FixelCourses/blob/master/MachineLearningMethods/2024_06/EnvConda.yml).  
   Save it as `EnvConda.yml`.
3. Open `Conda Prompt (conda)`. You should see something like `(base) ...` on the terminal:
![](https://i.imgur.com/AGDV0WF.png)
4. Navigate to the folder where `EnvConda.yml` is located.
5. Run the command: `conda env create -file EnvConda.yml`. It should take a while.   
   Once it finishes oyu may see the list of the list of environments with `conda env list`.
6. Once it is finished, run `conda activate <EnvName>`. You should see `(base)` changes into `(EnvName)`.  
   The `<EnvName>` is defined as the field `name: <EnvName>` in the `EnvConda.yml`.
7. Open VS Code (Run `code` on command line).
8. Open the folder of the notebooks of the course.
9. Open a notebook and make sure you set the Python Kernel to match the `<EnvName>`.  
   The version of Python match the version in `EnvConda.yml`.

### Visual Studio Code (VS Code)

 1. Install the `Python` extension by **Microsoft**.
 2. Install the `Jupyter` extension by **Microsoft**.
 3. Install the `Pylance` extension by **Microsoft**.

## Final Project

Solve the problem:

$$ \arg \min_{\boldsymbol{x}} \frac{1}{2} {\left\| \boldsymbol{x} - \boldsymbol{y} \right\|}_{2}^{2} \quad \text{subject to} \quad {\left\| \boldsymbol{D} \boldsymbol{x} \right\|}_{1} \leq \lambda $$

Where $\boldsymbol{D}$ is the _forward_ finite differences operator.
