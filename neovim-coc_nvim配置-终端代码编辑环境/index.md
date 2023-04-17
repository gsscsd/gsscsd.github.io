# Neovim+Coc.nvim配置 终端代码编辑环境


> 目前公司常用的环境还是linux的服务器，经常需要使用vim进行写代码，经过探索Neovim + Coc.nvim非常适合。以下全部采用源码安装（相比之下，直接用各个发行版的命令安装会更加简单）。[参考博客](https://www.cnblogs.com/cniwoq/p/13272746.html)

<!--more-->

### Neovim 和插件安装
neovim：[下载地址](https://github.com/neovim/neovim/releases)
```shell
wget https://github.com/neovim/neovim/releases/download/v0.4.3/nvim-linux64.tar.gz
tar -zxvf nvim-linux64.tar.gz
```
然后把neovim路径下的bin加入到~/.bashrc，然后在source一下就算是成功了。
#### Neovim的基本配置
```shell
filetype plugin on
" 设置为双字宽显示，否则无法完整显示如:☆
set ambiwidth=double
set t_ut= " 防止vim背景颜色错误
set showmatch " 高亮匹配括号
set matchtime=1
set report=0
set ignorecase
set nocompatible
set noeb
set softtabstop=4
set shiftwidth=4
set nobackup
set autoread
set nocompatible
set nu "设置显示行号
set backspace=2 "能使用backspace回删
syntax on "语法检测
set ruler "显示最后一行的状态
set laststatus=2 "两行状态行+一行命令行
set ts=4
set expandtab
set autoindent "设置c语言自动对齐
set t_Co=256 "指定配色方案为256
" set mouse=a "设置可以在VIM使用鼠标
set selection=exclusive
" set selectmode=mouse,key
set tabstop=4 "设置TAB宽度
set history=1000 "设置历史记录条数   
" 配色方案
" let g:seoul256_background = 234
colo monokai
set background=dark
set shortmess=atl
" colorscheme desert
"共享剪切板
set clipboard+=unnamed 
set cmdheight=3
if version >= 603
     set helplang=cn
     set encoding=utf-8
endif
set fencs=utf-8,ucs-bom,shift-jis,gb18030,gbk,gb2312,cp936
set termencoding=utf-8
set encoding=utf-8
set fileencodings=ucs-bom,utf-8,cp936
set fileencoding=utf-8
set updatetime=300
set shortmess+=c
set signcolumn=yes
 
" autocmd FileType json syntax match Comment +\/\/.\+$+
 
set foldmethod=indent " 设置默认折叠方式为缩进
set foldlevelstart=99 " 每次打开文件时关闭折叠
 
" hi Normal ctermfg=252 ctermbg=none "背景透明
" au FileType gitcommit,gitrebase let g:gutentags_enabled=0
if has("autocmd")
    au BufReadPost * if line("'\"") > 1 && line("'\"") <= line("$") | exe "normal! g'\"" | endif
endif
inoremap jj <Esc> "将jj映射到Esc
```
### 插件安装
#### Vim-Plug
目前由于vim插件越来越多，由此也出现了插件管理工具，目前用得比较广泛的有：Vundle，vim-plug和dein,推荐使用vim-plug(对Vundle更好的替换)，关于dein，引用某巨佬的原话：“过度设计”。
```shell
sh -c 'curl -fLo "${XDG_DATA_HOME:-$HOME/.local/share}"/nvim/site/autoload/plug.vim --create-dirs https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim'
```
这样，这个`~/.local/share/nvim/site/autoload/plug.vim`就会在你的目录下，并且vim会被调用。
创建nvim的配置文件（这个配置文件和vim的`.vimrc`）一样：
```shell
mkdir ~/.config/nvim/
nvim ~/.config/nvim/init.vim
```
然后把
```shell
call plug#begin('~/.vim/plugged')
call plug#end()
```
加入到`init.vim`中，这样以后在`call begin`和`call end`之间加上插件就可以使用了。

**之后的每个插件在init.vim文件中配置好后，要进行保存退出，再次进入nvim,使用命令 :PlugInstall安装**

#### indentLine
`indentLine`此插件提供的一个可视化的缩进，把`Plug 'Yggdroot/indentLine'`，放到`init.vim的call begin和call end`之间，同时加入一些简单的配置：
```shell
let g:indent_guides_guide_size            = 1  " 指定对齐线的尺寸
let g:indent_guides_start_level           = 2  " 从第二层开始可视化显示缩进
```
#### vim-monokai
`vim-monokai`，这个插件是nvim的一个主题，monokai这个配色是比较经典好看的主题。

在`call begin 和 call end`之间加上`Plug 'crusoexia/vim-monokai'`，然后把
```shell
colo monokai
```
加到`init.vim`后面。
#### vim-airline
`vim-airline`给`nvim`提供一个强大的状态栏和标签栏，当打开多个文本时，可以用它进行快速的切换，是一个很强大的工具。

在`call begin 和 call end`之间加上：
```shell
Plug 'vim-airline/vim-airline'       
Plug 'vim-airline/vim-airline-themes' "airline 的主题
```
然后在`init.vim`里加上一些个性的配置：
```shell
" 设置状态栏
let g:airline#extensions#tabline#enabled = 1
let g:airline#extensions#tabline#left_alt_sep = '|'
let g:airline#extensions#tabline#buffer_nr_show = 0
let g:airline#extensions#tabline#formatter = 'default'
let g:airline_theme = 'desertink'  " 主题
let g:airline#extensions#keymap#enabled = 1
let g:airline#extensions#tabline#buffer_idx_mode = 1
let g:airline#extensions#tabline#buffer_idx_format = {
       \ '0': '0 ',
       \ '1': '1 ',
       \ '2': '2 ',
       \ '3': '3 ',
       \ '4': '4 ',
       \ '5': '5 ',
       \ '6': '6 ',
       \ '7': '7 ',
       \ '8': '8 ',
       \ '9': '9 '
       \}
" 设置切换tab的快捷键 <\> + <i> 切换到第i个 tab
nmap <leader>1 <Plug>AirlineSelectTab1
nmap <leader>2 <Plug>AirlineSelectTab2
nmap <leader>3 <Plug>AirlineSelectTab3
nmap <leader>4 <Plug>AirlineSelectTab4
nmap <leader>5 <Plug>AirlineSelectTab5
nmap <leader>6 <Plug>AirlineSelectTab6
nmap <leader>7 <Plug>AirlineSelectTab7
nmap <leader>8 <Plug>AirlineSelectTab8
nmap <leader>9 <Plug>AirlineSelectTab9
" 设置切换tab的快捷键 <\> + <-> 切换到前一个 tab
nmap <leader>- <Plug>AirlineSelectPrevTab
" 设置切换tab的快捷键 <\> + <+> 切换到后一个 tab
nmap <leader>+ <Plug>AirlineSelectNextTab
" 设置切换tab的快捷键 <\> + <q> 退出当前的 tab
nmap <leader>q :bp<cr>:bd #<cr>
" 修改了一些个人不喜欢的字符
if !exists('g:airline_symbols')
    let g:airline_symbols = {}
endif
let g:airline_symbols.linenr = "CL" " current line
let g:airline_symbols.whitespace = '|'
let g:airline_symbols.maxlinenr = 'Ml' "maxline
let g:airline_symbols.branch = 'BR'
let g:airline_symbols.readonly = "RO"
let g:airline_symbols.dirty = "DT"
let g:airline_symbols.crypt = "CR" 
```
#### rainbow
`rainbow`是一个提供嵌套括号高亮的一个工具。
在`call begin 和 call end`之间加上`Plug 'luochen1990/rainbow'`
简单的修复相应的配置：
```shell
let g:rainbow_active = 1
let g:rainbow_conf = {
\   'guifgs': ['darkorange3', 'seagreen3', 'royalblue3', 'firebrick'],
\   'ctermfgs': ['lightyellow', 'lightcyan','lightblue', 'lightmagenta'],
\   'operators': '_,_',
\   'parentheses': ['start=/(/ end=/)/ fold', 'start=/\[/ end=/\]/ fold', 'start=/{/ end=/}/ fold'],
\   'separately': {
\       '*': {},
\       'tex': {
\           'parentheses': ['start=/(/ end=/)/', 'start=/\[/ end=/\]/'],
\       },
\       'lisp': {
\           'guifgs': ['darkorange3', 'seagreen3', 'royalblue3', 'firebrick'],
\       },
\       'vim': {
\           'parentheses': ['start=/(/ end=/)/', 'start=/\[/ end=/\]/', 'start=/{/ end=/}/ fold', 'start=/(/ end=/)/ containedin=vimFuncBody', 'start=/\[/ end=/\]/ containedin=vimFuncBody', 'start=/{/ end=/}/ fold containedin=vimFuncBody'],
\       },
\       'html': {
\           'parentheses': ['start=/\v\<((area|base|br|col|embed|hr|img|input|keygen|link|menuitem|meta|param|source|track|wbr)[ >])@!\z([-_:a-zA-Z0-9]+)(\s+[-_:a-zA-Z0-9]+(\=("[^"]*"|'."'".'[^'."'".']*'."'".'|[^ '."'".'"><=`]*))?)*\>/ end=#</\z1># fold'],
\       },
\       'css': 0,
\   }
\}
```
#### nerdtree
`nerdtree`是一个树形的目录管理插件，可以方便在nvim中进行当前文件夹中的文件切换

在`call begin 和 call end`之间加上
```shell
Plug 'preservim/nerdtree'
Plug 'Xuyuanp/nerdtree-git-plugin'
```
进行以下的简单配置
```shell
" autocmd vimenter * NERDTree  "自动开启Nerdtree
let g:NERDTreeWinSize = 25 "设定 NERDTree 视窗大小
let NERDTreeShowBookmarks=1  " 开启Nerdtree时自动显示Bookmarks
"打开vim时如果没有文件自动打开NERDTree
" autocmd vimenter * if !argc()|NERDTree|endif
"当NERDTree为剩下的唯一窗口时自动关闭
autocmd bufenter * if (winnr("$") == 1 && exists("b:NERDTree") && b:NERDTree.isTabTree()) | q | endif
" 设置树的显示图标
let g:NERDTreeDirArrowExpandable = '+'
let g:NERDTreeDirArrowCollapsible = '-'
let NERDTreeIgnore = ['\.pyc$']  " 过滤所有.pyc文件不显示
let g:NERDTreeShowLineNumbers=0 " 是否显示行号
let g:NERDTreeHidden=0     "不显示隐藏文件
""Making it prettier
let NERDTreeMinimalUI = 1
let NERDTreeDirArrows = 1
nnoremap <F3> :NERDTreeToggle<CR> " 开启/关闭nerdtree快捷键
```
默认的情况下是关闭的，可以把autocmd vimenter * NERDTree的注释打开，改成默认开启的状态。同时把开关的按键映射到了F3，按下可以打开或关闭。
#### vim-cpp-enhanced-highlight
`vim-cpp-enhanced-highlight`，这个是用来加强C++的高亮的
```shell
Plug 'octol/vim-cpp-enhanced-highlight'
```
#### vim-snippets
`honza/vim-snippets`用来提供C++的snippets.
```shell
Plug 'honza/vim-snippets'
```
#### coc.nvim
`coc.nvim`是目前非常流行的，集代码补全、静态检测、函数跳转等功能的一个引擎。

安装`coc.nvim`，需要安装`nodejs`。和`nvim`一样，这个也有编译好的版本，解压后把bin的路径放到放到`.bashrc`里，之后运行
```shell
npm install -g neovim
```
在`init.vim`安装
```shell
Plug 'neoclide/coc.nvim', {'branch': 'release'}
```
以下是coc的默认配置：
```shell
" if hidden is not set, TextEdit might fail.
set hidden
" Some servers have issues with backup files, see #649
set nobackup
set nowritebackup
 
" You will have bad experience for diagnostic messages when it's default 4000.
set updatetime=300
 
" don't give |ins-completion-menu| messages.
set shortmess+=c
 
" always show signcolumns
set signcolumn=yes
 
" Use tab for trigger completion with characters ahead and navigate.
" Use command ':verbose imap <tab>' to make sure tab is not mapped by other plugin.
inoremap <silent><expr> <TAB>
      \ pumvisible() ? "\<C-n>" :
      \ <SID>check_back_space() ? "\<TAB>" :
      \ coc#refresh()
inoremap <expr><S-TAB> pumvisible() ? "\<C-p>" : "\<C-h>"
 
function! s:check_back_space() abort
  let col = col('.') - 1
  return !col || getline('.')[col - 1]  =~# '\s'
endfunction
 
" Use <c-space> to trigger completion.
inoremap <silent><expr> <c-space> coc#refresh()
 
" Use <cr> to confirm completion, `<C-g>u` means break undo chain at current position.
" Coc only does snippet and additional edit on confirm.
inoremap <expr> <cr> pumvisible() ? "\<C-y>" : "\<C-g>u\<CR>"
" Or use `complete_info` if your vim support it, like:
" inoremap <expr> <cr> complete_info()["selected"] != "-1" ? "\<C-y>" : "\<C-g>u\<CR>"
 
" Use `[g` and `]g` to navigate diagnostics
nmap <silent> [g <Plug>(coc-diagnostic-prev)
nmap <silent> ]g <Plug>(coc-diagnostic-next)
" Remap keys for gotos
nmap <silent> gd <Plug>(coc-definition)
nmap <silent> gy <Plug>(coc-type-definition)
nmap <silent> gi <Plug>(coc-implementation)
nmap <silent> gr <Plug>(coc-references)
 
" Use K to show documentation in preview window
nnoremap <silent> K :call <SID>show_documentation()<CR>
 
function! s:show_documentation()
  if (index(['vim','help'], &filetype) >= 0)
    execute 'h '.expand('<cword>')
  else
    call CocAction('doHover')
  endif
endfunction
 
" Highlight symbol under cursor on CursorHold
autocmd CursorHold * silent call CocActionAsync('highlight')
 
" Remap for rename current word
nmap <leader>rn <Plug>(coc-rename)
 
" Remap for format selected region
xmap <leader>f  <Plug>(coc-format-selected)
nmap <leader>f  <Plug>(coc-format-selected)
 
augroup mygroup
  autocmd!
  " Setup formatexpr specified filetype(s).
  autocmd FileType typescript,json setl formatexpr=CocAction('formatSelected')
  " Update signature help on jump placeholder
  autocmd User CocJumpPlaceholder call CocActionAsync('showSignatureHelp')
augroup end
 
" Remap for do codeAction of selected region, ex: `<leader>aap` for current paragraph
xmap <leader>a  <Plug>(coc-codeaction-selected)
nmap <leader>a  <Plug>(coc-codeaction-selected)
 
" Remap for do codeAction of current line
nmap <leader>ac  <Plug>(coc-codeaction)
" Fix autofix problem of current line
nmap <leader>qf  <Plug>(coc-fix-current)
 
" Create mappings for function text object, requires document symbols feature of languageserver.
xmap if <Plug>(coc-funcobj-i)
xmap af <Plug>(coc-funcobj-a)
omap if <Plug>(coc-funcobj-i)
omap af <Plug>(coc-funcobj-a)
 
" Use `:Format` to format current buffer
command! -nargs=0 Format :call CocAction('format')
 
" Use `:Fold` to fold current buffer
command! -nargs=? Fold :call     CocAction('fold', <f-args>)
 
" use `:OR` for organize import of current buffer
command! -nargs=0 OR   :call     CocAction('runCommand', 'editor.action.organizeImport')
```

#### CocVim

对于Python，直接装anaconda就行了，同时需要用pip装一下pynvim。
```shell
pip install pynvim
pip install -U jedi-language-server --use-feature=2020-resolver
```
在`nvim`输入`:checkhealth`，会进行基本环境的检查，保证你用的环境不会出问题即可.
`coc.nvim `提供了很多有用的扩展工具在`coc extensions`中，这些扩展在nvim中通过命令`:CocInstall`进行安装
```shell
:CocInstall coc-clangd # C++环境插件
:CocInstall coc-cmake  # Cmake 支持 
:CocInstall coc-git    # git 支持
:CocInstall coc-highlight  # 高亮支持
:CocInstall coc-jedi   # jedi
:CocInstall coc-java   # java
:CocInstall coc-json   # json 文件支持
:CocInstall coc-python # python 环境支持
:CocInstall coc-sh     # bash 环境支持
:CocInstall coc-snippets # python提供 snippets
:CocInstall coc-vimlsp # lsp
:CocInstall coc-yaml   # yaml
:CocInstall coc-pairs  # 自动补全括号
```
在snaipppts的提示中，可以通过 Ctrl + j 进行移动修改（参考https://github.com/neoclide/coc-snippets)。
在函数里，可以通过按下`gd`进行函数的跳转，`ctrl + o `回调到上次的位置。

### 附录--个人配置
```shell
call plug#begin('~/.vim/plugged') " 记得先创建该目录
    Plug 'honza/vim-snippets'
    Plug 'sainnhe/sonokai'
    Plug 'vim-python/python-syntax'
    " Plug 'vim-scripts/ctags.vim'
    Plug 'crusoexia/vim-monokai'
    Plug 'tmux-plugins/vim-tmux-focus-events'
    Plug 'vim-airline/vim-airline'
    Plug 'scrooloose/nerdcommenter'
    Plug 'vim-airline/vim-airline-themes'
    Plug 'luochen1990/rainbow'
    Plug 'octol/vim-cpp-enhanced-highlight'
    Plug 'Yggdroot/indentLine'
    Plug 'preservim/nerdtree'
    Plug 'Xuyuanp/nerdtree-git-plugin'
    Plug 'neoclide/coc.nvim', {'branch': 'release'}
    " Plug 'majutsushi/tagbar'
    Plug 'lifepillar/vim-solarized8'
call plug#end()

let mapleader = ","  " set map leader
filetype plugin on
" 设置为双字宽显示，否则无法完整显示如:☆
set ambiwidth=double
set t_ut= " 防止vim背景颜色错误
set showmatch " 高亮匹配括号
set matchtime=1
set noshowmode " block mode display
set novisualbell noerrorbells
set report=0
set ignorecase
set cursorline "highlight current line
set noeb
set softtabstop=2
set shiftwidth=2
set nobackup
set autoread
set nocompatible
set nu "设置显示行号
set backspace=2 "能使用backspace回删
syntax on "语法检测
set ruler "显示最后一行的状态
set laststatus=2 "两行状态行+一行命令行
" set ts=4
set expandtab
set autoindent "设置c语言自动对齐
set t_Co=256 "指定配色方案为256
" set mouse=a "设置可以在VIM使用鼠标
set selection=exclusive
" set selectmode=mouse,key
set tabstop=2 "设置TAB宽度
set history=1000 "设置历史记录条数
set shortmess=atl
set clipboard+=unnamed
set cmdheight=1
if version >= 603
    set helplang=cn
    set encoding=utf-8
endif
set fencs=utf-8,ucs-bom,shift-jis,gb18030,gbk,gb2312,cp936
set termencoding=utf-8
set encoding=utf-8
set fileencodings=ucs-bom,utf-8,cp936
" set updatetime=300
set shortmess+=c
set signcolumn=yes
" reset cursor when vim exits
au VimLeave * set guicursor=a:ver25-blinkon0

" +================================ 可视化缩进 =====================================+ "
" set notermguicolors
" let g:sonokai_style = 'shusia'
" let g:sonokai_enable_italic = 1
" let g:sonokai_disable_italic_comment = 1
" colo sonokai
colo monokai
set background=dark
hi CursorLine ctermfg=NONE ctermbg=NONE cterm=NONE
hi Normal ctermfg=NONE ctermbg=234 guifg=NONE
hi LineNr ctermfg=yellow ctermbg=236 cterm=NONE
" hi Terminal ctermfg=NONE ctermfg=NONE cterm=NONE
hi EndOfBuffer ctermbg=NONE
" visual mode bg
set foldmethod=indent " 设置默认折叠方式为缩进
set foldlevelstart=99 " 每次打开文件时关闭折叠

" hi Normal ctermfg=252 ctermbg=none "背景透明
" au FileType gitcommit,gitrebase let g:gutentags_enabled=0
if has("autocmd")
    au BufReadPost * if line("'\"") > 1 && line("'\"") <= line("$") | exe "normal! g'\"" | endif
endif

" neovim python set
let g:python3_host_prog = "/Users/gsscsd/miniconda3/bin/python"

let g:python_highlight_all = 1
" +================================ 可视化缩进 =====================================+ "

let g:indent_guides_enable_on_vim_startup = 0  " 默认关闭
let g:indent_guides_guide_size            = 1  " 指定对齐线的尺寸
let g:indent_guides_start_level           = 2  " 从第二层开始可视化显示缩进

" +================================== NERDTree =======================================+ "
" autocmd vimenter * NERDTree  "自动开启Nerdtree
let g:NERDTreeWinSize = 25 "设定 NERDTree 视窗大小
let NERDTreeShowBookmarks=1  " 开启Nerdtree时自动显示Bookmarks
"打开vim时如果没有文件自动打开NERDTree
" autocmd vimenter * if !argc()|NERDTree|endif
"当NERDTree为剩下的唯一窗口时自动关闭
autocmd bufenter * if (winnr("$") == 1 && exists("b:NERDTree") && b:NERDTree.isTabTree()) | q | endif
" 设置树的显示图标
let g:NERDTreeDirArrowExpandable = '+'
let g:NERDTreeDirArrowCollapsible = '-'
let g:NERDTreeShowLineNumbers=0 " 是否显示行号
let g:NERDTreeHidden=0     "不显示隐藏文件
""Making it prettier
let NERDTreeMinimalUI = 1
let NERDTreeDirArrows = 1
" autocmd vimenter *  NERDTreeToggle
let NERDTreeIgnore = ['\.pyc$', '__pycache__']
map <leader>nn :NERDTreeToggle<cr>
map <leader>nb :NERDTreeFromBookmark<Space>
map <leader>nf :NERDTreeFind<cr>

" set key map
nnoremap <silent> <S-n> :set nu!<CR> " 开关行号
" +================================== 按键映射 =======================================+ "

" self key map:
" <leader>s : open key
" <leader>d : close key
" <leader>e : norm key
inoremap <silent> jj <Esc> "将jj映射到Esc
nmap <leader>sp :set paste<CR>i
nmap <leader>dp :set nopaste<CR>
nnoremap ; :
xnoremap ; :
nnoremap j gj
nnoremap k gk
nnoremap ^ g^
nnoremap $ g$
nnoremap <leader>dh :noh<CR>

" +================================== coc.nvim  ======================================+ "
" if hidden is not set, TextEdit might fail.
set hidden
" Some servers have issues with backup files, see #649
set nobackup
set nowritebackup

" You will have bad experience for diagnostic messages when it's default 4000.
set updatetime=300

" don't give |ins-completion-menu| messages.
set shortmess+=c

" always show signcolumns
set signcolumn=yes

" Use tab for trigger completion with characters ahead and navigate.
" Use command ':verbose imap <tab>' to make sure tab is not mapped by other plugin.
inoremap <silent><expr> <TAB>
      \ pumvisible() ? "\<C-n>" :
      \ <SID>check_back_space() ? "\<TAB>" :
      \ coc#refresh()
inoremap <expr><S-TAB> pumvisible() ? "\<C-p>" : "\<C-h>"

function! s:check_back_space() abort
  let col = col('.') - 1
  return !col || getline('.')[col - 1]  =~# '\s'
endfunction

" Use <c-space> to trigger completion.
inoremap <silent><expr> <c-space> coc#refresh()

" Use <cr> to confirm completion, `<C-g>u` means break undo chain at current position.
" Coc only does snippet and additional edit on confirm.
inoremap <expr> <cr> pumvisible() ? "\<C-y>" : "\<C-g>u\<CR>"
" Or use `complete_info` if your vim support it, like:
" inoremap <expr> <cr> complete_info()["selected"] != "-1" ? "\<C-y>" : "\<C-g>u\<CR>"

" Use `[g` and `]g` to navigate diagnostics
nmap <silent> [g <Plug>(coc-diagnostic-prev)
nmap <silent> ]g <Plug>(coc-diagnostic-next)
" Remap keys for gotos
nmap <silent> gd <Plug>(coc-definition)
nmap <silent> gy <Plug>(coc-type-definition)
nmap <silent> gi <Plug>(coc-implementation)
nmap <silent> gr <Plug>(coc-references)

" Use K to show documentation in preview window
nnoremap <silent> K :call <SID>show_documentation()<CR>

function! s:show_documentation()
  if (index(['vim','help'], &filetype) >= 0)
    execute 'h '.expand('<cword>')
  else
    call CocAction('doHover')
  endif
endfunction

" Highlight symbol under cursor on CursorHold
autocmd CursorHold * silent call CocActionAsync('highlight')

" Remap for rename current word
nmap <leader>rn <Plug>(coc-rename)

" Remap for format selected region
xmap <leader>f  <Plug>(coc-format-selected)
nmap <leader>f  <Plug>(coc-format-selected)

augroup mygroup
  autocmd!
  " Setup formatexpr specified filetype(s).
  autocmd FileType typescript,json setl formatexpr=CocAction('formatSelected')
  " Update signature help on jump placeholder
  autocmd User CocJumpPlaceholder call CocActionAsync('showSignatureHelp')
augroup end

" Remap for do codeAction of selected region, ex: `<leader>aap` for current paragraph
xmap <leader>a  <Plug>(coc-codeaction-selected)
nmap <leader>a  <Plug>(coc-codeaction-selected)

" Remap for do codeAction of current line
nmap <leader>ac  <Plug>(coc-codeaction)
" Fix autofix problem of current line
nmap <leader>qf  <Plug>(coc-fix-current)

" Create mappings for function text object, requires document symbols feature of languageserver.
xmap if <Plug>(coc-funcobj-i)
xmap af <Plug>(coc-funcobj-a)
omap if <Plug>(coc-funcobj-i)
omap af <Plug>(coc-funcobj-a)

" Use `:Format` to format current buffer
command! -nargs=0 Format :call CocAction('format')

" Use `:Fold` to fold current buffer
command! -nargs=? Fold :call     CocAction('fold', <f-args>)

" use `:OR` for organize import of current buffer
command! -nargs=0 OR   :call     CocAction('runCommand', 'editor.action.organizeImport')
"
" inoremap <silent><expr> <TAB>
"       \ pumvisible() ? coc#_select_confirm() :
"       \ coc#expandableOrJumpable() ? \"\<C-r>=coc#rpc#request('doKeymap', ['snippets-expand-jump',''])\<CR>" :
"       \ <SID>check_back_space() ? \"\<TAB>" :
"       \ coc#refresh()
"
" function! s:check_back_space() abort
"   let col = col('.') - 1
"   return !col || getline('.')[col - 1]  =~# '\s'
" endfunction
"
" let g:coc_snippet_next = '<tab>'
" +=================================== tagbar =======================================+ "

let g:tagbar_width=30
" +================================== airline =======================================+ "
let g:airline#extensions#tabline#enabled = 1
let g:airline#extensions#tabline#left_alt_sep = '|'
let g:airline#extensions#tabline#buffer_nr_show = 0
let g:airline#extensions#tabline#formatter = 'default'
let g:airline_theme = 'desertink'
let g:airline#extensions#keymap#enabled = 1
let g:airline#extensions#tabline#buffer_idx_mode = 1
let g:airline#extensions#tabline#buffer_idx_format = {
       \ '0': '0 ',
       \ '1': '1 ',
       \ '2': '2 ',
       \ '3': '3 ',
       \ '4': '4 ',
       \ '5': '5 ',
       \ '6': '6 ',
       \ '7': '7 ',
       \ '8': '8 ',
       \ '9': '9 '
       \}
nmap <leader>1 <Plug>AirlineSelectTab1
nmap <leader>2 <Plug>AirlineSelectTab2
nmap <leader>3 <Plug>AirlineSelectTab3
nmap <leader>4 <Plug>AirlineSelectTab4
nmap <leader>5 <Plug>AirlineSelectTab5
nmap <leader>6 <Plug>AirlineSelectTab6
nmap <leader>7 <Plug>AirlineSelectTab7
nmap <leader>8 <Plug>AirlineSelectTab8
nmap <leader>9 <Plug>AirlineSelectTab9
nmap <leader>- <Plug>AirlineSelectPrevTab
nmap <leader>= <Plug>AirlineSelectNextTab
nmap <leader>q :bp<cr>:bd #<cr>
if !exists('g:airline_symbols')
    let g:airline_symbols = {}
endif
let g:airline_symbols.linenr = "CL" " current line
let g:airline_symbols.whitespace = ''
" let g:airline_left_sep = ']'
" let g:airline_left_alt_sep = ')'
" let g:airline_right_sep = '['
" let g:airline_right_alt_sep = '('
let g:airline_symbols.maxlinenr = 'Ml' "maxline
let g:airline_symbols.branch = 'BR'
let g:airline_symbols.readonly = "RO"
let g:airline_symbols.dirty = "DT"
let g:airline_symbols.crypt = "CR"
" +=============================== NERD Commenter ====================================+ "
"add spaces after comment delimiters by default
let g:NERDSpaceDelims = 1
au FileType python let g:NERDSpaceDelims = 0

" Use compact syntax for prettified multi-line comments
let g:NERDCompactSexyComs = 1

" Align line-wise comment delimiters flush left instead of following code indentation
let g:NERDDefaultAlign = 'left'

" Set a language to use its alternate delimiters by default
let g:NERDAltDelims_java = 1

" Add your own custom formats or override the defaults
" let g:NERDCustomDelimiters = { 'c': { 'left': '/**','right': '*/' } }

" Allow commenting and inverting empty lines (useful when commenting a region)
let g:NERDCommentEmptyLines = 1

" Enable trimming of trailing whitespace when uncommenting
let g:NERDTrimTrailingWhitespace = 1

" Enable NERDCommenterToggle to check all selected lines is commented or not
let g:NERDToggleCheckAllLines = 1


" +=============================== rainbow Parentheses ==============================+ "
" ((((((()))))))
let g:rainbow_active = 1
let g:rainbow_conf = {
\   'guifgs': ['#FFE66F', '#00FFFF', '#46A3FF', '#AAAAFF', '#FFB5B5'],
\   'ctermfgs': ['lightyellow', 'lightcyan','lightblue', 'lightmagenta'],
\   'operators': '_,_',
\   'parentheses': ['start=/(/ end=/)/ fold', 'start=/\[/ end=/\]/ fold', 'start=/{/ end=/}/ fold'],
\}
```
