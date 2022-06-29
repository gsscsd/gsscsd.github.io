# Eclipse Theia教程


> Theia 是 Eclipse 推出的云端和桌面 IDE 平台，完全开源。Theia 是基于 VS Code 开发的，它的模块化特性非常适合二次开发，比如华为云 CloudIDE、阿里云 Function Compute IDE 便是基于 Theia 开发。
> 
> 参考链接：
>
> - [打造你的 Could IDE](https://blog.hvnobug.com/post/remote-ide)
>
> - [随时随地敲代码，基于Theia快速部署自己的云开发环境](https://zhuanlan.zhihu.com/p/144866584)
>
> - [Theia官方文档](https://theia-ide.org/docs/composing_applications)

<!--more-->

## Theia前置软件
> 1. nodejs 10以上
> ```shell
> curl -o- https://raw.githubusercontent.com/creationix/nvm/v0.33.5/install.sh | bash
> nvm install 10
> ```
> 2. yarn
> ```shell
> npm install -g yarn
> ```
> 3. docker
> ```shell
> # centos 7版本
> sudo yum install -y yum-utils device-mapper-persistent-data lvm2
> sudo yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
> sudo yum install docker-ce docker-ce-cli containerd.io
> sudo service docker start
> sudo systemctl enable docker
> # or 脚本自动安装
> curl -fsSL get.docker.com -o get-docker.sh
> sudo sh get-docker.sh
> # 如果连接速度慢，可加上--mirror参数指定镜像，如 sudo sh get-docker.sh --mirror Aliyun
> ```

## 基于NodeJs版本
### 基础程序配置
> 新建一个目录：my-app
```shell
mkdir my-app
cd my-app
```
> 创建package.json文件
```shell
{
  "private": true,
  "dependencies": {
    "@theia/callhierarchy": "next",
    "@theia/file-search": "next",
    "@theia/git": "next",
    "@theia/json": "next",
    "@theia/markers": "next",
    "@theia/messages": "next",
    "@theia/mini-browser": "next",
    "@theia/navigator": "next",
    "@theia/outline-view": "next",
    "@theia/plugin-ext-vscode": "next",
    "@theia/preferences": "next",
    "@theia/preview": "next",
    "@theia/search-in-workspace": "next",
    "@theia/terminal": "next"
  },
  "devDependencies": {
    "@theia/cli": "next"
  },
  "scripts": {
    "prepare": "yarn run clean && yarn build && yarn run download:plugins",
    "clean": "theia clean",
    "build": "theia build --mode development",
    "start": "theia start --plugins=local-dir:plugins",
    "download:plugins": "theia download:plugins"
  },
  "theiaPluginsDir": "plugins",
  "theiaPlugins": {
    "vscode-builtin-css": "https://github.com/theia-ide/vscode-builtin-extensions/releases/download/v1.39.1-prel/css-1.39.1-prel.vsix",
    "vscode-builtin-html": "https://github.com/theia-ide/vscode-builtin-extensions/releases/download/v1.39.1-prel/html-1.39.1-prel.vsix",
    "vscode-builtin-javascript": "https://github.com/theia-ide/vscode-builtin-extensions/releases/download/v1.39.1-prel/javascript-1.39.1-prel.vsix",
    "vscode-builtin-json": "https://github.com/theia-ide/vscode-builtin-extensions/releases/download/v1.39.1-prel/json-1.39.1-prel.vsix",
    "vscode-builtin-markdown": "https://github.com/theia-ide/vscode-builtin-extensions/releases/download/v1.39.1-prel/markdown-1.39.1-prel.vsix",
    "vscode-builtin-npm": "https://github.com/theia-ide/vscode-builtin-extensions/releases/download/v1.39.1-prel/npm-1.39.1-prel.vsix",
    "vscode-builtin-scss": "https://github.com/theia-ide/vscode-builtin-extensions/releases/download/v1.39.1-prel/scss-1.39.1-prel.vsix",
    "vscode-builtin-typescript": "https://github.com/theia-ide/vscode-builtin-extensions/releases/download/v1.39.1-prel/typescript-1.39.1-prel.vsix",
    "vscode-builtin-typescript-language-features": "https://github.com/theia-ide/vscode-builtin-extensions/releases/download/v1.39.1-prel/typescript-language-features-1.39.1-prel.vsix"
  }
}
```
> 本质上，Theia应用程序和扩展包都是Node.js包。每一个包都包含一个package.json文件，里面列出了包的一些元数据，如name、version、运行时和构建时的依赖关系等。
> - name和version被省略了，因为我们不打算将它作为一个依赖项来使用。同时它被标记为private，因为不打算将它发布为一个独立的Node.js包。
> - 我们在dependencies中列出了所有运行时依赖的扩展包，如@theia/navigator。
> - 有些扩展包需要额外的工具来进行安装，例如，@theia/python需要Python Language Server来安装。此时你需要参考[Theia官方文档](https://theia-ide.org/docs/composing_applications)。
> - 我们将@theis/cli列为构建时的依赖项，它提供了构建和运行应用程序的脚本。
> - 在上面的package.json文件中，我们加入了vscode插件。

### 构建
> 首先安装所有依赖
> 
> ```shell
> yarn
> ```
> 其次使用Theia Cli编译程序
> ```shell
> yarn theia build
> ```
> yarn在我们的应用程序上下文中查找@theia/cli提供的theia可执行文件，然后使用theia执行build命令。 由于该应用程序默认情况下是在生产模式下构建的（即经过混淆和缩小），因此可能需要一段时间。

### 运行
> 对于程序，需要提供一个工作区路径作为第一个参数打开，并提供--hostname和--port选项以将应用程序部署在特定的网络接口和端口上，例如 在所有接口和端口8080上打开/workspace：
> ```shell
> yarn start /my-workspace --hostname 0.0.0.0 --port 8080
> ```

## 使用 Docker 构建
### 从 theia-apps 快速构建
> Theia提供了不同版本的镜像，可以在theia-apps 选择自己需要的语言版本，可以支持 C++/Go/Python/Java/PHP/Ruby等多种语言。最简单的方法，就是直接获取镜像启动容器。
> ```shell
> docker pull theiaide/theia-full
> docker run -it --init -p 3000:3000 -v "$(pwd):/home/project:cached" theiaide/theia-full:latest
> ```
> 其中，$(pwd) 代表的是将当前目录挂载到 Docker 容器中，也可以指定文件目录。
> 
> 然而，需要特别注意的是，Theia 本身没有认证机制，任何知道公网 IP 和端口号的人都可使用。因此，不推荐这种方法。

### 构建更安全的版本
> Theia-https-docker 增加了 token 认证和 https，可以在标准镜像中加入 security layer，强烈建议使用它构造自己的镜像。构建也非常简单，按以下三个步骤操作即可，其中第三步的 --build-arg app= 填入需要使用的语言版本，这里使用的也是 full 版本。
```shell
git clone https://github.com/theia-ide/theia-apps.git
cd theia-apps/theia-https-docker
docker build . --build-arg app=theia-full -t theiaide/theia-full-security
```
> 构建完成后，可以通过docke images查看镜像
```shell
docker run --init -itd -p 10443:10443 -e token=mysecrettoken -v "/home/coding:/home/project:cached" theiaide/theia-full-security
```
> token 后接的是访问口令,/home/coding是指定的目录，打开 https://ip地址:10443，输入 token 便可打开 Web IDE。也可直接使用 https://ip地址:10443/?token=mysecrettoken 直接打开。

### 解决权限问题
> 使用docker版的Theia，会发现Theia 无法写入文件。这是 Theia 默认使用了 1000 的 userid，跟宿主机不一样，从而造成的权限问题。
> 
> 解决方法有这么几个：
> 
> - 将挂载的文件权限改为 777，这种方法不太安全： chmod -R 777 /home/coding
> - 指定用户运行，但如果使用的是 root，仍会有些不安全：`docker run --user=root --init -it -p 10443:10443 -e token=mysecrettoken -v "/home/coding:/home/project:cached" theiaide/theia-full-security`
> - 将挂载的文件夹属主改为1000，推荐这种方法： chown -R 1000 /home/coding

