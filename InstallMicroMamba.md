# Fixel Courses - Install MicroMamba

This guide shows how to install [MicroMamba](https://github.com/mamba-org/mamba) on Windows computer.
The installation is configured to be _Portable_, hence does not affect any other configuration on the computer.

## Prerequisites

 - Windows 10 / Windows 11.
 - [Windows Terminal](https://github.com/microsoft/terminal) installed.  
   It is recommended to install it using [Windows Store - Terminal](https://apps.microsoft.com/detail/9n0dx20hk701).  
   It is recommended to set [_Windows Terminal_ as the default terminal](https://superuser.com/questions/1789185).  
 - The [VS Code](https://code.visualstudio.com) editor is installed and available on the system path variable.

## Install MicroMamaba Environment

1. `PythonMamba` Folder  
   Create a folder for the MicroMamba environment.  
   It is suggested to create it in `C:\Users\<UserName>\PythonMamba`.  
   Yet it may be created anywhere, as long all paths are adjusted accordingly.
2. Download MicroMamba  
    - Go to [MicroMamba Releases repository](https://github.com/mamba-org/micromamba-releases).  
    - Locate the latest release on the [Release Pages](https://github.com/mamba-org/micromamba-releases/releases).
    - Download the file for **Windows**. Its pattern should be `micromamba-win-64.exe`.  
      You may need to click `Show all <> assets`.
    - Put the file inside `C:\Users\<UserName>\PythonMamba`.
    - **Rename the file** into `micromamba.exe`.
3. Download MicroMamba Script  
   Download the file [`PythonMamba.ps1`](./PythonMamba.ps1) from the [repository root](./).  
   Put the file in `C:\Users\<UserName>\PythonMamba`.
4. Download the MicroMamba Icon  
   Download the file [`PythonMamba.png`](./PythonMamba.png) from the [repository root](./).  
   Put the file in `C:\Users\<UserName>\PythonMamba`.
5. Configure Windows Terminal
    - Open Windows Terminal.
    - Open the settings tab.
    - Click on `Add new profile`. Choose `New empty profile`.
    - Set the `Name` property to `PythonMamba`.
    - Set the `Command line` property to `%SystemRoot%\\System32\\WindowsPowerShell\\v1.0\\powershell.exe -ExecutionPolicy Bypass -NoExit -File "<PathPythonMamba>\PythonMamba.ps1"`.  
      Change the `<PathPythonMamba>` to the folder path.
    - Set the `Icon` property to `"<PathPythonMamba>\PythonMamba.png"`.  
      You may chose `Browse` and navigate manually.
    - Click `Save`.
6. Open `MambaPython`  
   Run the `PythonMamba` profile from _Windows Terminal_.
7. Validate the Environment  
   Run `micromamba --version` to verify installation.  
   From now on auto complete using `Tab` will be available.

The end game of the process is a single folder with 3 files: `micromamba.exe`, `PythonMamba.ps1` and `PythonMamba.png`.  
Then create a Windows Terminal profile which executes the `PythonMamba.ps1` script.

## Install Conda Environment

The `micromamba` command can replace `conda` in most commands.  
Yet there are subtle differences as described in [Micromamba User Guide](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html).  

For instance, to install a `conda` environment from file, use `micromamba create -f <PathToFile>`.  
See [Fixel Courses - Install Conda Environment](./InstallCondaEnv.md).  
Replace `conda` with `micromamba` with the note about the `create` case as above.


## Remarks

### The Command Line

The _Windows Terminal_ profile runs the following command:

```cmd
%SystemRoot%\\System32\\WindowsPowerShell\\v1.0\\powershell.exe -ExecutionPolicy Bypass -NoExit -File "<PathPython>\MambaPython.ps1" 
```

The `-ExecutionPolicy Bypass` parameter should allow the file to run.  
In case the system does not run the script, the system policy should be changed to allow running scripts.  
See [`Set-ExecutionPolicy`](https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.security/set-executionpolicy).  

1. Open `Windows PowerShell` profile as Administrator.
2. Run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`.


### Images of the Installation Phase

**Remark**: Naming may be different from above. Stick to the above.

Downloading MicroMamba (Download the latest version available):

![](https://i.imgur.com/qF2aisQ.png)
<!-- ![](https://i.postimg.cc/658ngMKn/image.png) -->


<!-- ![](https://i.imgur.com/qd8SkzA.png) -->
<!-- ![](https://i.postimg.cc/jqpL1RvG/image.png) -->

<!-- ![](https://i.imgur.com/CEohdVS.png) -->
<!-- https://i.postimg.cc/j5GdBh6z/CEohdVS.png -->

The end game (3 files in the same folder, You may call the folder as you wish):

![](https://i.imgur.com/q6v4MgN.png)
<!-- ![](https://postimg.cc/YvrmGzD4/4b984d55) -->

The Button Down menu in _Windows Terminal_ for Settings or choosing a Profile:

![](https://i.imgur.com/X1HnfTS.png)
<!-- ![](https://i.postimg.cc/0NnSghr7/image.png) -->

The Settings page to add a new profile:

![](https://i.imgur.com/okGMJP2.png)
<!-- ![](https://i.postimg.cc/tJKZtjmG/image.png) -->

Run the new set profile:

![](https://i.imgur.com/ebYB4v1.png)
<!-- ![](https://i.postimg.cc/Y2W6nrQW/image.png) -->
