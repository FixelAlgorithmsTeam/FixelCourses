
# General Tips & Tricks working on remote

[![Visitors](https://img.shields.io/badge/Fixel-Algorithms-blue)](https://fixelalgorithms.gitlab.io)
[![Visitors](https://api.visitorbadge.io/api/combined?path=https%3A%2F%2Fgithub.com%2FRoyiAvital%2FStackExchangeCodes&labelColor=%23f47373&countColor=%23555555&style=plastic)](https://github.com/FixelAlgorithmsTeam/FixelCourses) <!-- https://www.visitorbadge.io -->

### Install zsh terminal on remote

reference: https://dev.to/hannahgooding/how-i-customise-my-terminal-with-oh-my-zsh-macos-427i


```
sudo apt install zsh

sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
```

restart session.

### Install Auto Suggestion & Syntax highlighting ZSH Plugins

```
git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions
```

```
git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting
```


Now go and edit `~/.zshrc` by running `sudo nano ~/.zshrc` and locate the line starts with `PLUGINS` and replace with 

```
plugins=(git zsh-autosuggestions zsh-syntax-highlighting web-search)
```

restart session you should now have auto complete and syntax highlights
