# 配置hexo支持latex公式


### 第一步： 安装Kramed

```shell
npm uninstall hexo-renderer-marked --save
npm install hexo-renderer-kramed --save
```

<!--more-->

### 第二步：更改文件配置

> 打开`/node_modules/hexo-renderer-kramed/lib/renderer.js`,更改：

```js
// Change inline math rule
function formatText(text) {
    // Fit kramed's rule: $$ + \1 + $$
    return text.replace(/`\$(.*?)\$`/g, '$$$$$1$$$$');
}

为，直接返回text

// Change inline math rule
function formatText(text) {
    return text;
}
```

### 第三步: 停止使用 hexo-math，并安装mathjax包

```shell
npm uninstall hexo-math --save
npm install hexo-renderer-mathjax --save
```

### 第四步: 更新 Mathjax 的 配置文件

>  打开`/node_modules/hexo-renderer-mathjax/mathjax.html` ，注释掉第二个`<script>`

```js
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_CHTML"></script>
```

### 第五步: 更改默认转义规则

> 打开`/node_modules\kramed\lib\rules\inline.js` 

```js
escape: /^\\([\\`*{}\[\]()#$+\-.!_>])/,
更改为
escape: /^\\([`*\[\]()# +\-.!_>])/,
    
em: /^\b_((?:__|[\s\S])+?)_\b|^\*((?:\*\*|[\s\S])+?)\*(?!\*)/,
更改为
em: /^\*((?:\*\*|[\s\S])+?)\*(?!\*)/,
```

### 第六步: 开启mathjax

> 打开你所使用主题的`_config.yml`文件

```yaml
mathjax:
    enable: true
```

### 最后的最后

在每个文章的开头添加

```yaml
mathjax: true
```

例如:

```yaml
title: tensorflow实例与线性回归
date: 2018-12-29 15:16:08
mathjax: true
tags:
	- python
	- 深度学习
	- tensorflow
categories: 深度学习
```


