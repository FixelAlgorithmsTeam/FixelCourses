# ************
# Version 1.1.000	31/10/2026	Royi Avital
# - Validate the local micromamba executable and shell hook before reporting success.
# - Use portable user folders with consistent home environment variables.
# - Build paths with JoinPaths and return to the parent folder.
# Version 1.0.000	26/01/2024	Royi Avital
# - First release version.
# ************

if ([string]::IsNullOrWhiteSpace($env:WT_SESSION)) {
    Write-Warning 'This script is intended to be launched from a Windows Terminal PowerShell profile.'
}

# Joins path components, similar to Python's `os.path.join()`.
function JoinPaths {
    param (
        [Parameter(Mandatory = $true, ValueFromRemainingArguments = $true)]
        [string[]] $PathItems
    )
    [System.IO.Path]::Combine($PathItems)
}

$MicromambaPath = JoinPaths $PSScriptRoot 'micromamba.exe'
if (-not (Test-Path -LiteralPath $MicromambaPath -PathType Leaf -ErrorAction Stop)) {
    throw "Could not find micromamba.exe beside PythonMamba.ps1: $MicromambaPath. Put the executable in the same folder as this script."
}

# Creating Folders
$UserFolder          = JoinPaths $PSScriptRoot 'User'
$UserLocalAppData    = JoinPaths $UserFolder 'AppData' 'Local'
$UserRoamingAppData  = JoinPaths $UserFolder 'AppData' 'Roaming'
$UserDocuments       = JoinPaths $UserFolder 'Documents'
$UserProgramData     = JoinPaths $UserFolder 'ProgramData'
$UserPublic          = JoinPaths $UserFolder 'Public'
$UserFolders = @(
    $UserFolder
    $UserLocalAppData
    $UserRoamingAppData
    $UserDocuments
    $UserProgramData
    $UserPublic
)
New-Item -ItemType Directory -Path $UserFolders -Force -ErrorAction Stop | Out-Null

# Setting Env Variables
$env:ALLUSERSPROFILE = $UserProgramData
$env:APPDATA = $UserRoamingAppData
$env:HOMEDRIVE = [System.IO.Path]::GetPathRoot($UserFolder).TrimEnd([System.IO.Path]::DirectorySeparatorChar)
$env:HOMEPATH = $UserFolder.Substring($env:HOMEDRIVE.Length)
$env:LOCALAPPDATA = $UserLocalAppData
$env:ProgramData = $UserProgramData
$env:Public = $UserPublic
$env:USERPROFILE = $UserFolder
$env:MAMBA_ROOT_PREFIX = $PSScriptRoot

# Invoke the hook
$Hook = & $MicromambaPath shell hook -s powershell | Out-String
if ($LASTEXITCODE -ne 0) {
    throw "micromamba shell hook failed with exit code $LASTEXITCODE."
}
if ([string]::IsNullOrWhiteSpace($Hook)) {
    throw 'micromamba returned an empty PowerShell hook.'
}
Invoke-Expression -Command $Hook -ErrorAction Stop
# The PowerShell environment does not need the `activate` command

Set-Location -LiteralPath (Split-Path -LiteralPath $PSScriptRoot -Parent) -ErrorAction Stop

Write-Host 'Finished setting Conda (MicroMamba) environment!'