# opencv-python安装


<center>前言</center>

> 在图像处理和计算机视觉领域，opencv是个好工具，这学期刚学完图像处理，正好，用opencv把常用的算法调用一下，学习一下图像处理。

<!--more-->

####  安装软件

1.首先说明，电脑是win8.1，32位系统

2.安装方法：

```python 
pip install opencv-python
```

如果报以下错误：

```python 
ImportError: DLL load failed: 找不到指定的程序。
```

那么，你有以下几个方法来解决：

1. 去微软官网下载[vsredist.exe](https://www.microsoft.com/en-us/download/details.aspx?id=48145),下载你电脑对应的版本号。
2. 去[python库](https://www.lfd.uci.edu/~gohlke/pythonlibs/)下载whl

#### 运行例子

```python 
# coding=utf-8

import cv2
import numpy as np


def main():
    x = np.ones((512, 512), np.float)
    cv2.imshow("Demo", x)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
```

可以看到一个白色的窗口出现。

![](http://owzdb6ojd.bkt.clouddn.com/17-12-4/61206048.jpg)
