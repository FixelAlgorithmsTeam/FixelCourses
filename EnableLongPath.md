# Fixel Courses - Enable Long Path on Windows Computer

[![](./FixelAlgorithmsLogo.png)](https://fixelalgorithms.gitlab.io)

[![Visitors](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FRoyiAvital%2FStackExchangeCodes&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=Visitors+%28Daily+%2F+Total%29&edge_flat=false)](https://github.com/FixelAlgorithmsTeam/FixelCourses)
[![Visitors](https://api.visitorbadge.io/api/combined?path=https%3A%2F%2Fgithub.com%2FRoyiAvital%2FStackExchangeCodes&labelColor=%23f47373&countColor=%23555555&style=plastic)](https://github.com/FixelAlgorithmsTeam/FixelCourses) <!-- https://www.visitorbadge.io -->

This guide shows how to enable Long Path Support on Windows (See [Maximum Path Length Limitation](https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation)).

> [!NOTE]
> Enabling the feature requires _Admin_ privileges on the system.

## Checking the Long Path Feature Status

In order to check if the feature is enabled do as following:

1. Open Windows Terminal  
   Search for `Terminal` on Windows Menu (_Start_).
2. PowerShell  
   Run the Windows PowerShell Profile (Default in most cases).
3. Run Registry Query  
   Run `(Get-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem").LongPathsEnabled`.  
   If the answer is `1` the feature is enabled. If the value is `0` the feature must be enabled.


![](https://i.imgur.com/AVfWC6P.png)
<!-- ![](https://i.postimg.cc/wTtSmwDr/Check-Long-Path-Support.png) -->

In the case above, the feature is disabled.

## Enable Long Path Support

There are 2 main options to enable the feature:

 - Using PowerShell.
 - Using Windows Settings.

Both will require `Administrator Rights` on the machine.

#### Enable Long Path Support by Windows Settings

> [!NOTE]
> This method is verified on Windows 11.

1. Open Windows Settings  
   On Start Menu search for settings.
2. Advanced  
   Got to `System -> Advanced`.
3. Enable Long Path  
   Turn `Enable long path` to `On`.

![](https://i.imgur.com/x7pmhAE.png)
<!-- ![](https://i.postimg.cc/2mxvMzn5/Enable-Long-Path-Windows-Settings.png) -->

#### Enable Long Path Support by Windows PowerShell

1. Open PowerShell as Administrator  
   On Windows Terminal click a `Right Click` on the PowerShell Profile.  
   Choose `Run As Adminstrator`.
2. Enable Long Path Support  
   Run the command `New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force`.
3. Verify  
   Run `(Get-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem").LongPathsEnabled`.  
   The return value should be `1`. 

![](https://i.imgur.com/EhwBsDm.png)
<!-- ![](https://i.postimg.cc/QtFcNKW4/Run-As-Admin.png) -->

![](https://i.imgur.com/Fquxu8b.png)
<!-- ![](https://i.postimg.cc/WznNQX1D/Enable-Long-Path.png) -->

![](https://i.imgur.com/JTuTs09.png)
<!-- ![](https://i.postimg.cc/50mW0jdD/Verify-Long-Path-Support.png) -->