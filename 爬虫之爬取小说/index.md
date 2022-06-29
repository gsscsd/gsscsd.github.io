# 爬虫之爬取小说


##  <center >前言</center>

> 好久没写博客了，懒死了，为了培养同学对python的喜爱，顺手写了一个爬虫，也是对自己知识的巩固，其实，这个爬虫挺简单的，半小时就能写完，但是2.x的python，各种字符编码的转换，真心累，而且，转到3.x系列还很多bug，暂时就先这样吧，等有时间再优化。

<!--more-->

> 这份代码，定向爬取了奇书网的女频言情小说（其实，我应该爬仙侠小说的），代码里面有注释，所及，就这样了。

#### show code：

```python
# coding: utf-8

import requests
from bs4 import BeautifulSoup
import os
import multiprocessing
import urllib


'''
说明：
需要安装beautifulsoup4,lxml,requests
直接在cmd命令行下运行：
pip install beautifulsoup4
pip install lxml
pip install requests

开发环境为python 2.7.13
'''

def Schedule(a,b,c):
    '''''
    a:已经下载的数据块
    b:数据块的大小
    c:远程文件的大小
   '''
    per = 100.0 * a * b / c
    if per > 100 :
        per = 100
    print('%.2f%%' % per)

def has_class_but_no_target(tag):
    '''
    tag:筛选tag函数
    '''
    return tag.has_attr('href') and not tag.has_attr('target')

def get_all_download_link(url,data):
    '''
    url:小说详情页面，例如：https://www.qisuu.com/36457.html
    data:小说存放位置
    '''
#     s = requests.Session()
    response = requests.get(url)
    response.encoding = 'utf-8'
    html = response.text
    soup = BeautifulSoup(html,'lxml')
    div_list = soup.find('div',class_ = 'showDown')
    a_list = div_list.find_all('a')
    for k,v in enumerate(a_list):         
        href = v.get('href').encode('utf-8')
        if 'txt' in href:
            title = v.get('title')
            data = os.path.join(data,title + '.txt')
            print(title)
            ###下载小说
            urllib.urlretrieve(href,data,Schedule)

def get_all_link(url):
    '''
    url:每个主页面,例如:https://www.qisuu.com/soft/sort03/index_2.html
    '''
    s = requests.Session()
    response = s.get(url)
    response.encoding = 'utf-8'
    # print(response.text)
    html = response.text
    soup = BeautifulSoup(html,'lxml')
    div_list = soup.find('div',class_ = 'listBox')
    a_list = div_list.find_all(has_class_but_no_target)
    # print(a_list)
    a_all = []
    for a in a_list:
        link = a.get('href')
        if not 'soft' in link:
            a_all.append(link)
    return a_all

def main(url_links,base_url,data):
    '''
    url_links:所有的主页面的list
    base_url:基础url，做拼接用
    data:小说存放位置
    '''
    print('start runing:')
    all_links = []
    for url in url_links:
        all_link = get_all_link(url)
        all_links = all_links + all_link
    all_links = [base_url + link for link in all_links]
#     pool = multiprocessing.Pool(multiprocessing.cpu_count())
    for  i,link in enumerate(all_links):
        print('第{0}本小说正在下载：'.format(i + 1))
        get_all_download_link(link,data)
#         pool.apply_async(get_all_download_link, (url,data))
        print('第{0}本小说下载完成.'.format(i + 1))
    print('所有的小说下载完成。')

if __name__ == "__main__":
    url = 'https://www.qisuu.com/soft/sort03/'
    base_url = 'https://www.qisuu.com'
    ##小说存放的地方
    data = '../data'
    if not os.path.exists(data):
        os.mkdir(data)
    ##此处可以修改，设置爬取多少个页面，每个页面有15个小说，最大值为50
    maxNum = 2
    number = ['index_' + str(i) + '.html' for i in range(1,maxNum)]
    ##修正第一页的url
    number[0] = 'index.html'
    url_links = [url + i for i in number]
    main(url_links,base_url,data)
```

####  最后来张效果图

![效果图](http://owzdb6ojd.bkt.clouddn.com/17-10-21/38115755.jpg)
