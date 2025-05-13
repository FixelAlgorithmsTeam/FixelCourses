# Fixel Courses - Install Conda Environment

[![](./FixelAlgorithmsLogo.png)](https://fixelalgorithms.gitlab.io)

[![Visitors](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FRoyiAvital%2FStackExchangeCodes&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=Visitors+%28Daily+%2F+Total%29&edge_flat=false)](https://github.com/FixelAlgorithmsTeam/FixelCourses)
[![Visitors](https://api.visitorbadge.io/api/combined?path=https%3A%2F%2Fgithub.com%2FRoyiAvital%2FStackExchangeCodes&labelColor=%23f47373&countColor=%23555555&style=plastic)](https://github.com/FixelAlgorithmsTeam/FixelCourses) <!-- https://www.visitorbadge.io -->

The [`conda` package manager](https://en.wikipedia.org/wiki/Conda_(package_manager)) is an open source, cross platform, language agnostic package manager and environment management system.  
It is (2024) a rather popular package manager in the Python eco system.

## Install the Course `conda` Environment

This guide is mainly Windows users though the main steps can be replicated in `macOS` and `Linux`.  
It guides how to install a `conda` environment given in the form of a [YAML File](https://en.wikipedia.org/wiki/YAML).

Guides:

 - [Getting Started with `conda`](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html).
 - [The `conda` User Guide](https://docs.conda.io/projects/conda/en/latest/user-guide/index.html).
 - [The `conda` Cheat Sheet](https://docs.conda.io/projects/conda/en/latest/user-guide/cheatsheet.html).

### Auxiliary Steps

1. Update Windows  
   Make sure your Windows version is updated using `Windows Update`.   
   For Windows 10 it should be `Windows 10 Version 22H2` or later.
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
   The operation requires elevated privileges (`Administrator`).
   
### Install the Conda Environment

This is a basic instruction how to install the course environment under Anaconda (Or any `conda` managed distribution).  
It is assumed that the Windows OS is used.

1. Install the [MiniForge](https://github.com/conda-forge/miniforge/releases/latest) distribution of `conda`.  
   The package to download (Windows) should be named: `Miniforge<Version>-<YY>.<MM>.<Build>-<Patch>-Windows-x86_64.exe`.   
   On _17/12/2024_ the latest version is `Miniforge3-24.11.0-0-Windows-x86_64.exe`.   
   Use the option to install it per user. Do not register it as the default Python.  
   **Attention**: Do not install version named `pypy`.
2. Download the environment specification file.  
   Each course has its own file.  
   For instance, for `AI Program 2024_12` it is given by [`AIProgram\2024_12\EnvConda.yml`](https://github.com/FixelAlgorithmsTeam/FixelCourses/blob/master/AIProgram/2024_12/EnvConda.yml) (Link to GitHub page).  
   Save it as `EnvConda.yml`.
3. Open `Conda Prompt (conda)` / `MiniForge Prompt` from Windows Menu.    
   If `Mini Forge` is installed, look for `Mini Forge Prompt` on Windows Menu.  
   If `Mini  Conda` is installed, look for `Conda Prompt` or `Anaconda Prompt`.  
   You should see something like `(base) ...` on the terminal:
![](https://i.imgur.com/AGDV0WF.png)
4. Navigate to the folder where `EnvConda.yml` is located.
5. Run the command: `conda env create --file EnvConda.yml` (Equivalent to `conda env create -f EnvConda.yml`).  
   It will try to solve the dependency graph (It might take a while) and show the packages to be installed for approval.  
   Once approved, it will download the packages and install them.  
   **Remark**: In general, the command for arbitrary path is: `conda env create --file <Path\To\EnvConda.yml>`.  
   **Remark**: In MicroMamba no need to use `env`: `micromamba create --file <Path\To\EnvConda.yml>` / `micromamba create -f <Path\To\EnvConda.yml>`.  
6. Run `conda env list` to see the list of available environments.  
   Locate the environment of the specific course. This approves the previous step succeeded.
6. Run `conda activate <EnvName>` to activate an environment.  
   You should see `(base)` changes into `(EnvName)`.  
   The `<EnvName>` is defined as the field `name: <EnvName>` in the `EnvConda.yml`.
7. Open VS Code.  
   Run `code` on command line when the course environment is activated.  
   Launching from the activated environment allows `VS Code` inherit the system variables of the environment.
8. Open the folder of the notebooks of the course with `File -> Open Folder...`.
9. Open a notebook and make sure you set the Python Kernel to match the `<EnvName>`.  
   The version of Python match the version in `EnvConda.yml`.

### Visual Studio Code (VS Code)

 1. Install the `Python` extension by **Microsoft**.
 2. Install the `Jupyter` extension by **Microsoft**.
 2. Install the `Data Wrangler` extension by **Microsoft**.

