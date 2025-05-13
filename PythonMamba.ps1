# ************
# Version 1.0.000	26/01/2024	Royi Avital
# - First release version.
# ************

# Aux Variable
$UserFolder = $PSScriptRoot + "\User"

# Creating Folders
if (!(Test-Path -Path $UserFolder)) { mkdir $UserFolder }
if (!(Test-Path -Path "$UserFolder\AppData\Local")) { mkdir "$UserFolder\AppData\Local" }
# if (!(Test-Path -Path "$UserFolder\AppData\Local\Temp")) { mkdir "$UserFolder\AppData\Local\Temp" }
if (!(Test-Path -Path "$UserFolder\AppData\Roaming")) { mkdir "$UserFolder\AppData\Roaming" }
if (!(Test-Path -Path "$UserFolder\Documents")) { mkdir "$UserFolder\Documents" }
if (!(Test-Path -Path "$UserFolder\ProgramData")) { mkdir "$UserFolder\ProgramData" }
if (!(Test-Path -Path "$UserFolder\Public")) { mkdir "$UserFolder\Public" }

# Setting Env Variables
$env:ALLUSERSPROFILE = "$UserFolder\ProgramData"
$env:APPDATA = "$UserFolder\AppData\Roaming"
$env:HOMEPATH = $UserFolder
$env:LOCALAPPDATA = "$UserFolder\AppData\Local"
$env:ProgramData = "$UserFolder\ProgramData"
$env:Public = "$UserFolder\Public"
# $env:TEMP = "$UserFolder\AppData\Local\Temp"
# $env:TMP = "$UserFolder\AppData\Local\Temp"
$env:USERPROFILE = $UserFolder

# Mamba Settings
$env:MAMBA_ROOT_PREFIX = "$PSScriptRoot"

# Go to the `$env:MAMBA_ROOT_PREFIX` folder
Set-Location -Path "$env:MAMBA_ROOT_PREFIX"

# Assuming `micromamba.exe` is in `$env:MAMBA_ROOT_PREFIX`
if (!(Test-Path -Path "micromamba.exe")) {
	Write-Host "Could not find micromamba.exe, failed setting the environment!"
	exit
 }

# Invoke the hook
.\micromamba.exe shell hook -s powershell | Out-String | Invoke-Expression
# The PowerShell environment does not need the `activate` command

Write-Host "Finished setting Conda (MicroMamba) environment!"

Set-Location -Path "$PSScriptRoot\.."

exit