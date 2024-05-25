
# General Tips & Tricks working on remote

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
