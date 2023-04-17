# 爬虫之糗事百科

<center >前言</center>

>自己动手，丰衣足食。本来的想法是爬取糗事百科的段子，然后发送到自己的邮箱，这样可以在电脑上，定时运行爬虫，然后，在手机端看段子了。但是，qq邮箱的smtp，一直没有配置好，所以只能先把段子保存到本地，有时间在优化。

<!--more-->

##### talk is cheap,show my code:

```python
import requests
from bs4 import BeautifulSoup
import codecs
import os 

def bs_textAnalysize(content):
    soup = BeautifulSoup(content,'lxml')
    content_list = soup.find_all('div',class_ = 'content')
    text_list = []
    for text in content_list:
        span = text.find('span').text
        text_list.append(span)
    return text_list

def main(url,data):
    text_list = []
    for burl in url :
        response = requests.get(burl)
        response.encoding = 'utf-8'
        content = response.text
        text = bs_textAnalysize(content)
        text_list += text
        with codecs.open(data + '笑话.txt','w+','utf-8') as fp:
            for i,text in enumerate(text_list):
                fp.write('第'+ str(i + 1) + '个笑话:\n')
                fp.write(text + '\n')

if __name__ == '__main__':
    base_url = 'https://www.qiushibaike.com/text/page/'
    data = '../data/'
    if not os.path.exists(data):
        os.mkdir(data)
    maxNum = 10
    url = [base_url + str(i) for i in range(1,maxNum)]
    main(url,data)
    print('ok')
```

代码很简单，才36行。但是，代码爬取了，糗百的*text*分类下的10个页面的所有段子，大约有200多个。所以说，人生苦短，我用*python*。下面来一一解析。

#### 引入库

引入了*requests*和*BeautifulSoup*这两个最好用的第三方库，以及*python*自带的*os*和*codecs*库。

#### main函数

其中*url*是个*list*，里面全是糗事百科的连接，然后用*main*来调用其他函数，*main*有两个参数，一个是*url*的*list*，一个是笑话的存储位置（*data*）。

#### bs_textAnalysize 函数

使用*BeautifulSoup*来抽取段子的内容，然后将一个一个的段子加入到*list*中，并在*main*函数里，保存起来。

#### 最后来一张效果图

![photo](http://owzdb6ojd.bkt.clouddn.com/17-10-26/63642657.jpg)
