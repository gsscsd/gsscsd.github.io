# 多台电脑同时写hexo博客


---

1.找到github上面的博客仓库

> xxx.github.io   这个仓库的名字

2.在这个仓库上，新建一个分支，名字随便起，比如，我们起名为**hexo**

3.然后进入到设置里面，将**hexo**分支设为**默认分支**

<!--more-->

4.在本地新建一个文件夹，然后将xxx.github.io这个仓库clone下来

```shell
git clone git@github.com:xxx/xxxx.github.io.git
```

5.进入这个clone下来的仓库，此时这个仓库里面什么都没有，我们将原来的hexo项目里面的所有文件，复制到这个仓库里面，然后push到远程仓库上，**在push之前，进入themes文件夹下面，删除主题的.git文件夹**

```shell
git add .
git commit -m "xxxx"
git push
```

6.此时可以去远程仓库查看，上面已经全是hexo项目的源码了

7.在另一台电脑上面，新建一个文件夹，然后clone这个仓库

```shell
git clone git@github.com:xxx/xxxx.github.io.git
```

8.进入仓库，就能看到在其他电脑上创建的hexo项目了，然后就可以在这台电脑写博客，写完，git提交。

- 写博客并部署

```shell
hexo new "xxxx"
hexo g
hexo d
```

- 将代码上传到git上

```shell
git add .
git commit -m "xx"
git push
```

PS：记住每次写博客之前，先**git pull**拉取仓库，然后在写博客。
