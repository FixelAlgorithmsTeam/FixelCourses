# Fixel Courses - Install Conda Environment

The [`conda` package manager](https://en.wikipedia.org/wiki/Conda_(package_manager)) is an open source, cross platform, language agnostic package manager and environment management system.  
It is (2024) a rather popular package manager in the Python eco system.

## Install the Course `conda` Environment

This guide is mainly Windows users though the main steps can be replicated in `macOS` and `Linux`.  
It guides how to install a `conda` environment given in the form of a [YAML File](https://en.wikipedia.org/wiki/YAML).

### Auxiliary Steps

1. Update Windows  
   Make sure your Windows version is updated using `Windows Update`.   
   For Windows 10 it should be `Windows 10 Version 22H2` or more.
2. Install Windows Terminal  
   It suggested to install [Windows Terminal](https://github.com/microsoft/terminal).  
   If can be done, the best way is to install it using the Windows Store: [Windows Store - Windows Terminal](https://apps.microsoft.com/detail/9N0DX20HK701).  
   It can be installed by downloading the latest official release from GitHub and using `Add-AppPackage` on `PowerShell Terminal`.  
   Once installed, set it to be the default console application on Windows.  
   Updated versions of Windows 11 (23H2 and newer) have the Windows Terminal pre installed and defined as the default.  
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
2. Download the environment specification file.  
   Each course has its own file.  
   For instance, for `AI Program 2024_12` it is given by [`EnvConda.yml`](https://github.com/FixelAlgorithmsTeam/FixelCourses/blob/master/AIProgram/2024_12/EnvConda.yml).  
   Save it as `EnvConda.yml`.
3. Open `Conda Prompt (conda)` / `MiniForge Prompt` from Windows Menu.    
   If `Mini Forge` is installed, look for `Mini Forge Prompt` on Windows Menu.  
   If `Mini  Conda` is installed, look for `Conda Prompt`.  
   You should see something like `(base) ...` on the terminal:
![](https://i.imgur.com/AGDV0WF.png)
4. Navigate to the folder where `EnvConda.yml` is located.
5. Run the command: `conda env create -file EnvConda.yml`. It should take a while.   
   Once it finishes you may see the list of the list of environments with `conda env list`.
6. Once it is finished, run `conda activate <EnvName>`. You should see `(base)` changes into `(EnvName)`.  
   The `<EnvName>` is defined as the field `name: <EnvName>` in the `EnvConda.yml`.
7. Open VS Code.  
   Run `code` on command line when the course environment is activated.
8. Open the folder of the notebooks of the course.
9. Open a notebook and make sure you set the Python Kernel to match the `<EnvName>`.  
   The version of Python match the version in `EnvConda.yml`.

### Visual Studio Code (VS Code)

 1. Install the `Python` extension by **Microsoft**.
 2. Install the `Jupyter` extension by **Microsoft**.

