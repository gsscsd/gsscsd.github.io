# 数据结构之顺序表

## <center> 前言

*严蔚敏的《数据结构》书里面全是类c代码，是c也不是c，感觉好别扭，于是用c语言重新写一遍，遇到好些bug，记录一下。*

以下出自[百度百科](http://baike.baidu.com/link?url=zgd9T0p46VwdJ0a2X90o5TrTiKZWrs24DWidX8Y4lEoIb-gIrC2OAkLqR_eyu74MUpBnHJb61HyV7EdbSTJidrY6QuaM4eEerhOl3jOeff5hp3QDs4YR2CjGnakauChR)
>顺序表是在计算机内存中以数组的形式保存的线性表，是指用一组地址连续的存储单元依次存储数据元素的线性结构。线性表采用顺序存储的方式存储就称之为顺序表。顺序表是将表中的结点依次存放在计算机内存中一组地址连续的存储单元中。

<!--more-->

### 首先给出基本数据定义
``` c
这是宏定义及顺序表定义：
#define ElemType int //数据类型
#define MaxSize 100  //最大存储量

//定义顺序表的数据结构s
typedef struct 
{
    ElemType data[MaxSize];
    int length;
}SqList;
```
### 给出基本函数定义：
```c
//初始化顺序表
void InitSqLsit(SqList *list);
//获取指定位置的元素
ElemType GetElem(SqList list,int p);
//获取元素在顺序表中的位置
int locationElem(SqList list,ElemType e);
//插入元素在指定位置
int InsertElem(SqList *list,int p,ElemType e);
//删除指定位置的元素
int DeletElem(SqList *list,int p,ElemType *e);
//遍历顺序表
void Traverse(SqList list);
```
### 给出每个函数的具体实现：

  #### 初始化：
  ```c
  //初始化顺序表，无返回值；
void InitSqLsit(SqList *list)
{
    puts("初始化顺序表");
    list->length = 0;
}
  ```
  ####  获取元素位置：
 ```c
 /**
*获取元素在表中的位置，失败返回-1
**/
int locationElem(SqList list,ElemType e)
{
    int i;
    for( i = 0; i < list.length; i++)
    {
        if(e == list.data[i])
        {
            return i+1;   //实际位置要加1
        }       
    }
    return -1;
}
 ```
  ####  获取指定位置的元素：
```c
/**
*获取指定位置的元素,成功返回元素值，失败返回-1
**/
ElemType GetElem(SqList list,int p)
{
    if(p<0||p>list.length-1)
    {
        return -1;
    }
    else 
    {
        return list.data[p - 1];    //p要减1
    }

}
```
  #### 插入元素：
 ```c
 /**
*在表中插入元素，失败返回-1，成功返回1
**/
int InsertElem(SqList *list,int p,ElemType e)
{
    if(p<0||p>list->length||list->length==MaxSize-1)
    {
        return -1;                          //插入失败，返回-1
    }
    else 
    {
        puts("开始插入数据：");
        int i;
        for( i = list->length-1; i>=p-1; i--) //由于从0开始存储所以要减1
        {
            list->data[i+1] = list->data[i];
        }
        list->data[p-1] = e;    //这里也要减1
        list->length++;
        puts("插入数据完成。");
        return 1;
    }
    
}
 ```
  #### 删除元素：
 ```c
 /**
*删除指定元素，并保存在e中,成功返回1，失败返回-1
**/
int DeletElem(SqList *list,int p,ElemType *e)
{
    if(p<0||p>list->length)
    {
        return -1;
    }
    else 
    {
        puts("开始删除数据：");
        *e = list->data[p-1];
        int i;
        for( i = p-1; i < list->length; i++)
        {
            list->data[i] = list->data[i+1];
        }
        list->length--;
        puts("删除数据完成。");
        return 1; 
    } 
}
 ```
  #### 遍历顺序表：
 ```c
 //遍历顺序表
void Traverse(SqList list)
{
    int i = 0;
    puts("遍历顺序表");
    for(i = 0;i < list.length;i++)
        printf("%d\t",list.data[i]);
    putchar('\n');
}
 ```
  #### 最后给出main函数：
```c
int main(int argc, char const *argv[])
{
    int i = 0;
    ElemType e;
    SqList list;
    //初始化顺序表
    InitSqLsit(&list);
    //测试数据
    puts("随机插入4个数据：");
    for(i = 0; i < 4; i++)
    {
        scanf("%d",&e);
        list.data[i] = e;
        list.length++;
    }

    //插入数据测试
    InsertElem(&list,2,6);
    //删除数据测试
    DeletElem(&list,5,&e);
    Traverse(list);
    e = GetElem(list,3);
    printf("%d is lie in %d\n",3,e);
    printf("4 is lie in :%d\n",locationElem(list,4));
    return 0;
}
```
### 输出结果：
![logo](数据结构之顺序表/2017-03-13_141307.png)






## **以下附上全部代码：**
```c
/**
说明：顺序表的C语言实现
**/

#include <stdio.h>
#define ElemType int
#define MaxSize 100

//定义顺序表的数据结构s
typedef struct 
{
    ElemType data[MaxSize];
    int length;
}SqList;

//初始化顺序表
void InitSqLsit(SqList *list);
//获取指定位置的元素
ElemType GetElem(SqList list,int p);
//获取元素在顺序表中的位置
int locationElem(SqList list,ElemType e);
//插入元素在指定位置
int InsertElem(SqList *list,int p,ElemType e);
//删除指定位置的元素
int DeletElem(SqList *list,int p,ElemType *e);
//遍历顺序表
void Traverse(SqList list);

int main(int argc, char const *argv[])
{
    int i = 0;
    ElemType e;
    SqList list;
    //初始化顺序表
    InitSqLsit(&list);
    //测试数据
    puts("随机插入4个数据：");
    for(i = 0; i < 4; i++)
    {
        scanf("%d",&e);
        list.data[i] = e;
        list.length++;
    }

    //插入数据测试
    InsertElem(&list,2,6);
    //删除数据测试
    DeletElem(&list,5,&e);
    Traverse(list);
    e = GetElem(list,3);
    printf("%d is lie in %d\n",3,e);
    printf("4 is lie in :%d\n",locationElem(list,4));
    return 0;
}

//初始化顺序表，无返回值；
void InitSqLsit(SqList *list)
{
    puts("初始化顺序表");
    list->length = 0;
}

//遍历顺序表
void Traverse(SqList list)
{
    int i = 0;
    puts("遍历顺序表");
    for(i = 0;i < list.length;i++)
        printf("%d\t",list.data[i]);
    putchar('\n');
}
/**
*获取元素在表中的位置，失败返回-1
**/
int locationElem(SqList list,ElemType e)
{
    int i;
    for( i = 0; i < list.length; i++)
    {
        if(e == list.data[i])
        {
            return i+1;   //实际位置要加1
        }       
    }
    return -1;
}

/**
*获取指定位置的元素,成功返回元素值，失败返回-1
**/
ElemType GetElem(SqList list,int p)
{
    if(p<0||p>list.length-1)
    {
        return -1;
    }
    else 
    {
        return list.data[p - 1];    //p要减1
    }

}
/**
*在表中插入元素，失败返回-1，成功返回1
**/
int InsertElem(SqList *list,int p,ElemType e)
{
    if(p<0||p>list->length||list->length==MaxSize-1)
    {
        return -1;                          //插入失败，返回-1
    }
    else 
    {
        puts("开始插入数据：");
        int i;
        for( i = list->length-1; i>=p-1; i--) //由于从0开始存储所以要减1
        {
            list->data[i+1] = list->data[i];
        }
        list->data[p-1] = e;    //这里也要减1
        list->length++;
        puts("插入数据完成。");
        return 1;
    }
    
}
/**
*删除指定元素，并保存在e中,成功返回1，失败返回-1
**/
int DeletElem(SqList *list,int p,ElemType *e)
{
    if(p<0||p>list->length)
    {
        return -1;
    }
    else 
    {
        puts("开始删除数据：");
        *e = list->data[p-1];
        int i;
        for( i = p-1; i < list->length; i++)
        {
            list->data[i] = list->data[i+1];
        }
        list->length--;
        puts("删除数据完成。");
        return 1; 
    } 
}

```


 


