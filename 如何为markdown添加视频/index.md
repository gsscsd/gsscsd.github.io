# 如何为markdown添加视频


#### 第一种方法

参考自[这篇文章](http://www.muzixing.com/pages/2014/03/21/markdowntian-jia-shi-pin-jiao-cheng-idina-menzehe-caleb-hylesji-qing-dui-chang-let-it-go.html)

以下是通用的代码框架：前提是网站必须支持iframe

```
<iframe height=498 width=510 src="这里嵌入视频的来源，优酷，爱奇艺等" frameborder=0 allowfullscreen></iframe>
```

<!--more-->

这是个视频标题：

<iframe height=498 width=510 src="http://player.youku.com/embed/XNjcyMDU4Njg0" frameborder=0 allowfullscreen></iframe>

#### 第二种方法

使用H5的video代码插入：

```
<video id="video" controls="" preload="none" poster="http://media.w3.org/2010/05/sintel/poster.png">
      <source id="mp4" src="http://media.w3.org/2010/05/sintel/trailer.mp4" type="video/mp4">
      <source id="webm" src="http://media.w3.org/2010/05/sintel/trailer.webm" type="video/webm">
      <source id="ogv" src="http://media.w3.org/2010/05/sintel/trailer.ogv" type="video/ogg">
      <p>Your user agent does not support the HTML5 Video element.</p>
    </video>
```

<video id="video" controls="" preload="none" poster="http://media.w3.org/2010/05/sintel/poster.png"> <source id="mp4" src="http://media.w3.org/2010/05/sintel/trailer.mp4" type="video/mp4">
<p>Your user agent does not support the HTML5 Video element.</p>
</video>


