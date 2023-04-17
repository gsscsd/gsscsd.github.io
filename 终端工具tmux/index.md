# 终端工具Tmux


> Tmux 的全称是 Terminal MUtipleXer，及终端复用软件。顾名思义，它的主要功能就是在你关闭终端窗口之后保持进程的运行，此外 Tmux 的另一个重大功能就是分屏

<!--more-->

## Tmux的基本配置
```shell
# -----------------------------------------------------------------------------
# Tmux 基本配置 - 要求 Tmux >= 2.3
# 如果不想使用插件，只需要将此节的内容写入 ~/.tmux.conf 即可flow-qiso
# -----------------------------------------------------------------------------
# https://gist.github.com/ryerh/14b7c24dfd623ef8edc7nf
# C-b 和 VIM 冲突，修改 Prefix 组合键为 Control-X，按键距离近
set -g prefix C-x

set -g base-index         1     # 窗口编号从 1 开始计数
set -g display-panes-time 10000 # PREFIX-Q 显示编号的驻留时长，单位 ms
# set -g mouse              on    # 开启鼠标
set -g pane-base-index    1     # 窗格编号从 1 开始计数
set -g renumber-windows   on    # 关掉某个窗口后，编号重排
setw -g allow-rename      off   # 禁止活动进程修改窗口名
setw -g automatic-rename  off   # 禁止自动命名新窗口
setw -g mode-keys         vi    # 进入复制模式的时候使用 vi 键位（默认是 EMACS）

# ----- Windows -----
# Use vi styled keys for scrolling & copying
set-window-option -g mode-keys vi
# ----- Panes -----
# Key bindings for switching panes
bind -n M-h select-pane -L # left
bind -n M-l select-pane -R # right
bind -n M-k select-pane -U # up
bind -n M-j select-pane -D # down
# Key bindings for creating panes
bind-key 1 split-window -h # horizontal
bind-key 2 split-window -v # verticle

# Contents on the right of the status bar
set -g status-right "#[fg=magenta,bold] #{prefix_highlight}#[fg=red,bold]CPU: #{cpu_percentage} #[fg=blue]Battery: #{battery_percentage} #[fg=green]%a %Y:%m:%d %H:%M:%S "
set -g status-interval 1 # refresh every second
set -g status-right-length 100 # maximum length for the right content of the status bar

# Contents on the left of the status bar
set -g status-left "#[fg=yellow,bold] ❐ #S   " # show the current session
set -g status-left-length 8 # maximum length for the left content of the status bar

#set -g default-terminal xterm-256color

# -----------------------------------------------------------------------------
# 使用插件 - via tpm
#   1. 执行 git clone https://github.com/tmux-plugins/tpm ~/.tmux/plugins/tpm
#   2. 执行 bash ~/.tmux/plugins/tpm/bin/install_plugins
# clone "Tmux Plugin Manager (TPM)" https://github.com/tmux-plugins/tpm.git ~/.tmux/plugins/tpm
# clone "tmux-battery" https://github.com/tmux-plugins/tmux-battery.git ~/.tmux/plugins/tmux-battery
# clone "tmux-cpu" https://github.com/tmux-plugins/tmux-cpu.git ~/.tmux/plugins/tmux-cpu
# clone "tmux-prefix-highlight" https://github.com/tmux-plugins/tmux-prefix-highlight.git ~/.tmux/plugins/
# -----------------------------------------------------------------------------

setenv -g TMUX_PLUGIN_MANAGER_PATH '~/.tmux/plugins'

# 推荐的插件（请去每个插件的仓库下读一读使用教程）
set -g @plugin 'seebi/tmux-colors-solarized'
set -g @plugin 'tmux-plugins/tmux-pain-control'
set -g @plugin 'tmux-plugins/tmux-prefix-highlight'
set -g @plugin 'tmux-plugins/tmux-resurrect'
set -g @plugin 'tmux-plugins/tmux-sensible'
set -g @plugin 'tmux-plugins/tmux-yank'
set -g @plugin 'tmux-plugins/tpm'

# tmux-resurrect
set -g @resurrect-dir '~/.tmux/resurrect'

# tmux-prefix-highlight
set -g status-right '#{prefix_highlight} #H | %a %Y-%m-%d %H:%M'
set -g @prefix_highlight_show_copy_mode 'on'
set -g @prefix_highlight_copy_mode_attr 'fg=white,bg=blue'

# 初始化 TPM 插件管理器 (放在配置文件的最后)
run '~/.tmux/plugins/tpm/tpm'

# -----------------------------------------------------------------------------
# 结束
# -----------------------------------------------------------------------------
```
上面是相对比较简约的配置，直接copy复制到.tmux.conf文件中即可。
在这个配置中，`<prefix>`被改为了 `Ctrl + x`

## 常用的快捷键
```shell
Pane

<prefix> 1 在右侧添加 Pane

<prefix> 2 在下方添加 Pane

<prefix> 0 关闭 Pane

<prefix> o 在 Pane 之间切换

<prefix> H 向左扩大 Pane

<prefix> J 向下扩大 Pane

<prefix> K 向上扩大 Pane

<prefix> L 向右扩大 Pane

<prefix> m 最大化/还原 Pane

<prefix> h/j/k/l 在 Pane 之间切换

Window

<prefix> c 创建新 Window

<prefix> <C-h> 切换至左侧 Window

<prefix> <C-l> 切换至右侧 Window

```shell
