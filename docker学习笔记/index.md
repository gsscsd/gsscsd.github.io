# Docker学习笔记


> 近期用到docker做实验，因此记录一下学习docker的笔记.
>
> 参考链接：
> - [一篇不一样的docker原理解析](https://zhuanlan.zhihu.com/p/22382728)
> - [一篇不一样的docker原理解析 提高篇](https://zhuanlan.zhihu.com/p/22403015)
> - [Docker 核心技术与实现原理](http://dockone.io/article/2941)
> - [终于有人把 Docker 讲清楚了，万字详解！](https://zhuanlan.zhihu.com/p/269485082?utm_source=wechat_session)
> - [docker 镜像基本原理和核心概念](https://zhuanlan.zhihu.com/p/108409686?utm_source=wechat_session)


<!--more-->

## docker的前世今生
docker是一个开源的应用容器引擎，基于Go语言开发的一种虚拟化技术，可以让开发者打包他们的应用以及依赖包到一个轻量级、可移植的容器中，然后发布到任何流行的linux服务器。docker底层是一种LXC技术，在LXC的基础之上，docker提供了一系列更强大的功能。
> LXC为Linux Container的简写。可以提供轻量级的虚拟化，以便隔离进程和资源，而且不需要提供指令解释机制以及全虚拟化的其他复杂性。相当于C++中的NameSpace。容器有效地将由单个操作系统管理的资源划分到孤立的组中，以更好地在孤立的组之间平衡有冲突的资源使用需求。
> 
> 与传统虚拟化技术相比，它的优势在于：
> - 与宿主机使用同一个内核，性能损耗小；
> - 不需要指令级模拟；
> - 不需要即时(Just-in-time)编译；
> - 容器可以在CPU核心的本地运行指令，不需要任何专门的解释机制；
> - 避免了准虚拟化和系统调用替换中的复杂性；
> - 轻量级隔离，在隔离的同时还提供共享机制，以实现容器与宿主机的资源共享。
> - 总结：Linux Container是一种轻量级的虚拟化的手段。
> 
> Linux Container提供了在单一可控主机节点上支持多个相互隔离的server container同时执行的机制。Linux Container有点像chroot，提供了一个拥有自己进程和网络空间的虚拟环境，但又有别于虚拟机，因为lxc是一种操作系统层次上的资源的虚拟化。

### 传统虚拟机概念
> 广义来说，虚拟机是一种模拟系统，即在软件层面上通过**模拟硬件**的输入和输出，让虚拟机的操作系统得以运行在没有物理硬件的环境中（也就是宿主机的操作系统上），其中能够模拟出硬件输入输出，让虚拟机的操作系统可以启动起来的程序，被叫做hypervisor。
> 
> 传统虚拟机的核心技术是通过软件模拟硬件，从而让虚拟机的操作系统运行起来，**当宿主机的硬件架构和虚拟硬件架构一致时，不模拟硬件输入输出，只是做下真实硬件输入输出的搬运工，那么虚拟机的指令执行速度，就可以和宿主机一致了**，而当宿主机的硬件架构和虚拟机硬件架构不一致时，需要**模拟硬件的输入输出**，此时虚拟机启动和运行速度相对就比较慢了。
> 
> 经典的例子：
> - 在linux系统上可以通过vm或者virtual软件安装Windows系统，虚拟windows系统速度和原生系统速度差不多
> - 在linux或者Windows系统上模拟android系统，则很慢，原因是android架构和windows的架构不一致

### 容器化概念
> 一般来说，虚拟机都会有自己的kernel，自己的硬件，这样虚拟机启动的时候需要先做开机自检，启动kernel，启动用户进程等一系列行为，虽然现在电脑运行速度挺快，但是这一系列检查做下来，也要几十秒，也就是虚拟机需要几十秒来启动。
>
> 计算机科学家发现其实我们创建虚拟机也不一定需要模拟硬件的输入和输出，假如宿主机和虚拟机他们的kernel是一致的，就不用做硬件输入输出的搬运工了，只需要做kernel输入输出的搬运工即可，为了有别于硬件层面的虚拟机，这种虚拟机被命名为 操作系统层虚拟化，也被叫做容器。

> 在虚拟机的系统中，虚拟机认为自己有独立的文件系统，进程系统，内存系统，等等一系列，所以为了让容器接近虚拟机，也需要有独立的文件系统，进程系统，内存系统，等等一系列，为了达成这一目的，主机系统采用的办法是：只要隔离容器不让它看到主机的文件系统，进程系统，内存系统，等等一系列，那么容器系统就是一个接近虚拟机的玩意了

## docker的核心技术
### Namespaces
> 命名空间（namespaces）是 Linux 为我们提供的用于分离进程树、网络接口、挂载点以及进程间通信等资源的方法。在日常使用 Linux 或者 macOS 时，我们并没有运行多个完全分离的服务器的需要，但是如果我们在服务器上启动了多个服务，这些服务其实会相互影响的，每一个服务都能看到其他服务的进程，也可以访问宿主机器上的任意文件，这是很多时候我们都不愿意看到的，我们更希望运行在同一台机器上的不同服务能做到完全隔离，就像运行在多台不同的机器上一样。
> 
> 在这种情况下，一旦服务器上的某一个服务被入侵，那么入侵者就能够访问当前机器上的所有服务和文件，这也是我们不想看到的，而 Docker 其实就通过 Linux 的 Namespaces 对不同的容器实现了隔离。
>
> Linux 的命名空间机制提供了以下七种不同的命名空间，包括 CLONE_NEWCGROUP、CLONE_NEWIPC、CLONE_NEWNET、CLONE_NEWNS、CLONE_NEWPID、CLONE_NEWUSER 和 CLONE_NEWUTS，通过这七个选项我们能在创建新的进程时设置新进程应该在哪些资源上与宿主机器进行隔离。

### 进程
> 进程是 Linux 以及现在操作系统中非常重要的概念，它表示一个正在执行的程序，也是在现代分时系统中的一个任务单元。在linux系统里面，有两个进程非常特殊，一个是 pid 为 1 的 /sbin/init 进程，另一个是 pid 为 2 的 kthreadd 进程，这两个进程都是被 Linux 中的上帝进程 idle 创建出来的，其中前者负责执行内核的一部分初始化工作和系统配置，也会创建一些类似 getty 的注册进程，而后者负责管理和调度其他的内核进程。

### 网络
> 如果 Docker 的容器通过 Linux 的命名空间完成了与宿主机进程的网络隔离，但是却有没有办法通过宿主机的网络与整个互联网相连，就会产生很多限制，所以 Docker 虽然可以通过命名空间创建一个隔离的网络环境，但是 Docker 中的服务仍然需要与外界相连才能发挥作用。
> 
> 每一个使用 docker run 启动的容器其实都具有单独的网络命名空间，Docker 为我们提供了四种不同的网络模式，Host、Container、None 和 Bridge 模式。
> 
> 在默认情况下，每一个容器在创建时都会创建一对虚拟网卡，两个虚拟网卡组成了数据的通道，其中一个会放在创建的容器中，会加入到名为 docker0 网桥中。docker0 会为每一个容器分配一个新的 IP 地址并将 docker0 的 IP 地址设置为默认的网关。网桥 docker0 通过 iptables 中的配置与宿主机器上的网卡相连，所有符合条件的请求都会通过 iptables 转发到 docker0 并由网桥分发给对应的机器。
> 
> Docker 通过 Linux 的命名空间实现了网络的隔离，又通过 iptables 进行数据包转发，让 Docker 容器能够优雅地为宿主机器或者其他容器提供服务。

### 挂载点
> 虽然我们已经通过 Linux 的命名空间解决了进程和网络隔离的问题，在 Docker 进程中我们已经没有办法访问宿主机器上的其他进程并且限制了网络的访问，但是 Docker 容器中的进程仍然能够访问或者修改宿主机器上的其他目录，这是我们不希望看到的。
> 
> 在新的进程中创建隔离的挂载点命名空间需要在 clone 函数中传入 CLONE_NEWNS，这样子进程就能得到父进程挂载点的拷贝，如果不传入这个参数子进程对文件系统的读写都会同步回父进程以及整个主机的文件系统。
> 
> 如果一个容器需要启动，那么它一定需要提供一个根文件系统（rootfs），容器需要使用这个文件系统来创建一个新的进程，所有二进制的执行都必须在这个根文件系统中。

### CGroups
> 我们通过 Linux 的命名空间为新创建的进程隔离了文件系统、网络并与宿主机器之间的进程相互隔离，但是命名空间并不能够为我们提供物理资源上的隔离，比如 CPU 或者内存，如果在同一台机器上运行了多个对彼此以及宿主机器一无所知的『容器』，这些容器却共同占用了宿主机器的物理资源。
> 
> 如果其中的某一个容器正在执行 CPU 密集型的任务，那么就会影响其他容器中任务的性能与执行效率，导致多个容器相互影响并且抢占资源。如何对多个容器的资源使用进行限制就成了解决进程虚拟资源隔离之后的主要问题，而 Control Groups（简称 CGroups）就是能够隔离宿主机器上的物理资源，例如 CPU、内存、磁盘 I/O 和网络带宽。
> 
> 每一个 CGroup 下面都有一个 tasks 文件，其中存储着属于当前控制组的所有进程的 pid，作为负责 cpu 的子系统，cpu.cfs_quota_us 文件中的内容能够对 CPU 的使用作出限制，如果当前文件的内容为 50000，那么当前控制组中的全部进程的 CPU 占用率不能超过 50%。
> 
> 如果系统管理员想要控制 Docker 某个容器的资源使用率就可以在 docker 这个父控制组下面找到对应的子控制组并且改变它们对应文件的内容，当然我们也可以直接在程序运行时就使用参数，让 Docker 进程去改变相应文件中的内容。
> 
> 当我们使用 Docker 关闭掉正在运行的容器时，Docker 的子控制组对应的文件夹也会被 Docker 进程移除，Docker 在使用 CGroup 时其实也只是做了一些创建文件夹改变文件内容的文件操作，不过 CGroup 的使用也确实解决了我们限制子容器资源占用的问题，系统管理员能够为多个容器合理的分配资源并且不会出现多个容器互相抢占资源的问题。

### 存储驱动
> 想要理解 Docker 使用的存储驱动，我们首先需要理解 Docker 是如何构建并且存储镜像的，也需要明白 Docker 的镜像是如何被每一个容器所使用的；Docker 中的每一个镜像都是由一系列只读的层组成的，Dockerfile 中的每一个命令都会在已有的只读层上创建一个新的层：
> ```shell
> FROM ubuntu:15.04
> COPY . /app
> RUN make /app
> CMD python /app/app.py
> ```

> 当镜像被 docker run 命令创建时就会在镜像的最上层添加一个可写的层，也就是容器层，所有对于运行时容器的修改其实都是对这个容器读写层的修改。
> 
>容器和镜像的区别就在于，所有的镜像都是只读的，而每一个容器其实等于镜像加上一个可读写的层，也就是同一个镜像可以对应多个容器。
![](http://dockone.io/uploads/article/20190625/6bfb353b9c10c0cbdbcd7fca18e92607.png)

### AUFS
> AUFS  即 Advanced UnionFS，作为联合文件系统，它能够将不同文件夹中的层联合（Union）到了同一个文件夹中，这些文件夹在 AUFS 中称作分支，整个『联合』的过程被称为联合挂载（Union Mount）。
![](http://dockone.io/uploads/article/20190625/15caedce18dad8d83e2ed3d02e29df03.png)
> 每一个镜像层或者容器层都是 /var/lib/docker/ 目录下的一个子文件夹；在 Docker 中，所有镜像层和容器层的内容都存储在 /var/lib/docker/aufs/diff/ 目录中。
> 而 /var/lib/docker/aufs/layers/ 中存储着镜像层的元数据，每一个文件都保存着镜像层的元数据，最后的 /var/lib/docker/aufs/mnt/ 包含镜像或者容器层的挂载点，最终会被 Docker 通过联合的方式进行组装。
![](http://dockone.io/uploads/article/20190625/e9ebb37a706756da7dcbd826fa613299.png)
> 上面的这张图片非常好的展示了组装的过程，每一个镜像层都是建立在另一个镜像层之上的，同时所有的镜像层都是只读的，只有每个容器最顶层的容器层才可以被用户直接读写，所有的容器都建立在一些底层服务（Kernel）上，包括命名空间、控制组、rootfs 等等，这种容器的组装方式提供了非常大的灵活性，只读的镜像层通过共享也能够减少磁盘的占用。

## docker架构

![](https://pic2.zhimg.com/80/v2-39f153682084c122b29d06e445c83855_1440w.jpg)

> docker daemon就是docker的守护进程即server端，可以是远程的，也可以是本地的，这个不是C/S架构吗，客户端Docker client 是通过rest api进行通信
>
> docker cli 用来管理容器和镜像，客户端提供一个只读镜像，然后通过镜像可以创建多个容器，这些容器可以只是一个RFS（Root file system根文件系统），也可以ishi一个包含了用户应用的RFS，容器再docker client中只是要给进程，两个进程之间互不可见
> 
> 用户不能与server直接交互，但可以通过与容器这个桥梁来交互，由于是操作系统级别的虚拟技术，中间的损耗几乎可以不计
> 
> 以 docker pull为例讲解，docker 的工作：docker client 组织配置和参数，把 pull 指令发送给 docker server，server 端接收到指令之后会交给对应的 handler。handler 会新开一个 CmdPull job 运行，这个 job 在 docker daemon 启动的时候被注册进来，所以控制权就到了 docker daemon 这边。docker daemon 是怎么根据传过来的 registry 地址、repo 名、image 名和tag 找到要下载的镜像呢？具体流程如下：
> 1. 获取 repo 下面所有的镜像 id：GET /repositories/{repo}/images
> 2. 获取 repo 下面所有 tag 的信息: GET /repositories/{repo}/tags
> 3. 根据 tag 找到对应的镜像 uuid，并下载该镜像
> - 获取该镜像的 history 信息，并依次下载这些镜像层: GET /images/{image_id}/ancestry
> - 如果这些镜像层已经存在，就 skip，不存在的话就继续
> - 获取镜像层的 json 信息：GET /images/{image_id}/json
> - 下载镜像内容： GET /images/{image_id}/layer
> - 下载完成后，把下载的内容存放到本地的 UnionFS 系统
> - 在 TagStore 添加刚下载的镜像信息

## docker使用

### docker配置
`vim /usr/lib/systemd/system/docker.service `
```yml
[Unit]
Description=Docker Application Container Engine
Documentation=http://docs.docker.com
After=network.target
Wants=docker-storage-setup.service
Requires=docker-cleanup.timer

[Service]
Type=notify
NotifyAccess=main
EnvironmentFile=-/run/containers/registries.conf
EnvironmentFile=-/etc/sysconfig/docker
EnvironmentFile=-/etc/sysconfig/docker-storage
EnvironmentFile=-/etc/sysconfig/docker-network
Environment=GOTRACEBACK=crash
Environment=DOCKER_HTTP_HOST_COMPAT=1
Environment=PATH=/usr/libexec/docker:/usr/bin:/usr/sbin
ExecStart=/usr/bin/dockerd-current --registry-mirror=https://rfcod7oz.mirror.aliyuncs.com \
          --add-runtime docker-runc=/usr/libexec/docker/docker-runc-current \
          --default-runtime=docker-runc \
          --exec-opt native.cgroupdriver=systemd \
          --userland-proxy-path=/usr/libexec/docker/docker-proxy-current \
          --init-path=/usr/libexec/docker/docker-init-current \
          --seccomp-profile=/etc/docker/seccomp.json \
          $OPTIONS \
          $DOCKER_STORAGE_OPTIONS \
          $DOCKER_NETWORK_OPTIONS \
          $ADD_REGISTRY \
          $BLOCK_REGISTRY \
          $INSECURE_REGISTRY \
          $REGISTRIES
ExecReload=/bin/kill -s HUP $MAINPID
LimitNOFILE=1048576
LimitNPROC=1048576
LimitCORE=infinity
TimeoutStartSec=0
Restart=on-abnormal
KillMode=process

[Install]
WantedBy=multi-user.target
```
如果更改存储目录就添加`--graph=/opt/docker`,如果要更改DNS`--dns=xxxx的方式指定`

### 简单入门命令
```shell
docker images # 查看已经下载的镜像
docker  save nginx >/tmp/nginx.tar.gz # 导出镜像
docker rmi -f nginx # 删除镜像
docker load </tmp/nginx.tar.gz # 导入镜像
docker ps # 查看容器
docker exec -it mynginx sh # 交互式进入容器
docker ps -a 　　　　　　#-a :显示所有的容器，包括未运行的
docker inspect mynginx # 查看容器详细信息
docker logs  -f mynginx  # -f  挂起这个终端，动态查看日志
```

### docker image 命令
```shell
docker images ：列出 docker host 机器上的镜像，可以使用 -f 进行过滤
docker build：从 Dockerfile 中构建出一个镜像
docker history：列出某个镜像的历史
docker import：从 tarball 中创建一个新的文件系统镜像
docker pull：从 docker registry 拉去镜像
docker push：把本地镜像推送到 registry
docker rmi： 删除镜像
docker save：把镜像保存为 tar 文件
docker search：在 docker hub 上搜索镜像
docker tag：为镜像打上 tag 标记
```
### 仓库相关
```shell
docker search $KEY_WORD              # 查找镜像
docker pull $REGISTRY:$TAG           # 获取镜像
docker push $IMAGE_NAME:$IMAGE_TAG   # 推送镜像到仓库，需要先登录
docker login $REGISTRY_URL           # 登录仓库
docker logout $REGISTRY_URL          # 退出仓库
docker info                          # 显示Docker详细的系统信息，可查看仓库地址
docker --help                        # 显示Docker的帮助信息
```

### 容器相关
```shell
docker attach $CONTAINER_ID  # 启动一个已存在的docker容器
docker stop $CONTAINER_ID    # 停止docker容器
docker start $CONTAINER_ID   # 启动docker容器
docker restart $CONTAINER_ID # 重启docker容器
docker kill $CONTAINER_ID    # 强制关闭docker容器
docker pause $CONTAINER_ID   # 暂停容器
docker unpause $CONTAINER_ID # 恢复暂停的容器
docker rename $CONTAINER_ID  # 重新命名docker容器
docker rm $CONTAINER_ID      # 删除容器
docker exec $CONTAINER_ID    # 运行已启动的容器 可以使用该命令配合-it进入容器交互执行

docker logs $CONTAINER_ID    # 查看docker容器运行日志，确保正常运行
docker inspect $CONTAINER_ID # 查看container的容器属性，比如ip等等
docker port $CONTAINER_ID    # 查看container的端口映射
docker top $CONTAINER_ID     # 查看容器中正在运行的进程
docker commit $CONTAINER_ID $NEW_IMAGE_NAME:$NEW_IMAGE_TAG # 将容器保存为镜像
docker ps -a                 # 查看所有容器
docker stats                 # 查看容器的资源使用情况

Ctrl+P+Q进行退出容器，正常退出不关闭容器，如果使用exit退出，那么在退出之后会关闭容器
```

### 镜像相关命令
```shell
docker images                      # 查看本地镜像
docker rmi $IMAGE_ID               # 删除本地镜像
docker inspect $IMAGE_ID           # 查看镜像详情
docker save $IMAGE_ID > 文件路径   # 保存镜像为离线文件
docker save -o 文件路径 $IMAGE_ID  # 保存镜像为离线文件
docker load < 文件路径             # 加载文件为docker镜像
docker load -i 文件路径            # 加载文件为docker镜像
docker tag $IMAGE_ID $NEW_IMAGE_NAME:$NEW_IMAGE_TAG  # 修改镜像TAG
docker run 参数 $IMAGE_ID $CMD     # 运行一个镜像
docker history $IMAGE_ID           # 显示镜像每层的变更内容
```

### docker run时参数
```shell
# -d，后台运行容器, 并返回容器ID；不指定时, 启动后开始打印日志, Ctrl+C退出命令同时会关闭容器
# -i，以交互模式运行容器, 通常与-t同时使用
# -t，为容器重新分配一个伪输入终端, 通常与-i同时使用
# --name container_name，设置容器名称, 不指定时随机生成
# -h container_hostname，设置容器的主机名, 默认随机生成
# --dns 8.8.8.8，指定容器使用的DNS服务器, 默认和宿主机一致
# -e docker_host=172.17.0.1，设置环境变量
# --cpuset="0-2" or --cpuset="0,1,2"，绑定容器到指定CPU运行
# -m 100M，设置容器使用内存最大值
# --net bridge，指定容器的网络连接类型, 支持bridge/host/none/container四种类型
# --ip 172.18.0.13，为容器指定固定IP（需要使用自定义网络none）
# --expose 8081 --expose 8082，开放一个端口或一组端口，会覆盖镜像设置中开放的端口
# -p [宿主机端口]:[容器内端口]，宿主机到容器的端口映射，可指定宿主机的要监听的IP，默认为0.0.0.0
# -P，注意是大写的, 宿主机随机指定一组可用的端口映射容器expose的所有端口
# -v [宿主机目录路径]:[容器内目录路径]，挂载宿主机的指定目录（或文件）到容器内的指定目录（或文件）
# --add-host [主机名]:[IP]，为容器hosts文件追加host, 默认会在hosts文件最后追加[主机名]:[容器IP]
# --volumes-from [其他容器名]，将其他容器的数据卷添加到此容器
# --link [其他容器名]:[在该容器中的别名]，添加链接到另一个容器，在本容器hosts文件中加入关联容器的记录，效果类似于--add-host
```

### Dockerfile
```shell
FROM ,  从一个基础镜像构建新的镜像
FROM ubuntu
MAINTAINER ,  维护者信息
MAINTAINER William <wlj@nicescale.com>
ENV ,  设置环境变量
ENV TEST 1
RUN ,  非交互式运行 shell 命令
RUN apt-get -y update
RUN apt-get -y install nginx
ADD ,  将外部文件拷贝到镜像里 ,src 可以为 url
ADD http://nicescale.com/ /data/nicescale.tgz
WORKDIR /path/to/workdir,  设置工作目录
WORKDIR /var/www
USER ,  设置用户 ID
USER nginx
VULUME <#dir>,  设置 volume
VOLUME [‘/data’]
EXPOSE ,  暴露哪些端口
EXPOSE 80 443
ENTRYPOINT [‘executable’, ‘param1’,’param2’] 执行命令
ENTRYPOINT ["/usr/sbin/nginx"]
CMD [“param1”,”param2”]
CMD ["start"]
```
