# 数据结构之链表


## <center> 前言
*今天来写一下C语言对链表的实现，由于在严蔚敏老师的书里，引入了c++语言的引用类型，在c语言中，并没有这一种类型，我用二级指针来代替。*

<!--more-->

### 基本定义
```c
#define ElemType int
#define MaxSize 100

//定义链表节点
typedef struct Lnode
{
    ElemType data;
    struct Lnode *next;
}LinkList;
```
### 函数原型声明
```c
//初始化链表
void InitList(LinkList **list);
//遍历链表
void Traverse(LinkList *list);
//查找指定位置元素
ElemType GetElem(LinkList *list,int p);
//在指定位置插入元素,失败返回-1，成功返回1
int InsertList(LinkList **list,int p,ElemType e);
//删除指定位置的元素,失败返回-1，成功返回1
int DeleteList(LinkList **list,int p,ElemType *e);
//链表是否为空，空返回1
int IsEmpty(LinkList *list);
//计算链表长度，返回链表长度
int LengthList(LinkList *list);
```
### 函数定义
#### 初始化函数
```c
//初始化链表
void InitList(LinkList **list)
{
    puts("InitList");
    int i = 0,n = 0;
    *list = malloc(sizeof(LinkList));
    LinkList *temp,*rNext;
    // (*list)->next = NULL;
    rNext = *list;
    puts("Input InitList Numbers:");
    scanf("%d",&n);
    for(i = 0; i < n; i++)
    {
        temp = malloc(sizeof(LinkList));
        puts("Input Number:");
        scanf("%d",&(temp->data));   
        //接下来有两种方法，头插，尾差，这里采用尾差
        rNext->next = temp;
        rNext = rNext->next;
    } 
    rNext->next = NULL;
}
```
#### 遍历函数
```c
//遍历链表
void Traverse(LinkList *list)
{
	puts("Begin Traverse:");
    LinkList *temp = list->next; //临时变量，方便遍历
    while(temp != NULL)
    {
        printf("%d\t",temp->data);
        temp = temp->next;
    }
    putchar('\n');
    puts("Finished Traverse.");

}
```
#### 查找函数
```c
//查找指定位置的元素
ElemType GetElem(LinkList *list,int p)
{
    printf("Find %d Elem:\n",p);
    if((p<0)||(p>LengthList(list)))
    {
        return -1;
    }
    else 
    {
        LinkList *rNext = list->next;
        for(int i = 0;i<p-1;i++)
        {
            rNext = rNext->next;
        }
        puts("End Find.");
        return rNext->data;
    }
}
```
#### 判空函数
```c
//链表是否为空，空返回1
int IsEmpty(LinkList *list)
{
    if(list->next == NULL)
    {
        return 1;
    }
    return 0;
}
```
#### 链表长度函数
```c
//计算链表长度，成功返回链表长度
int LengthList(LinkList *list)
{
    LinkList *rNext = list->next;
    int length = 0;
    while(rNext != NULL)
    {
        length++;
        rNext = rNext->next;
    }   
    return length;
}
```
#### 插入函数
```c
//在指定位置插入元素,失败返回-1，成功返回1
int InsertList(LinkList **list,int p,ElemType e)
{
    if((p<0)||(p>LengthList(*list)+1))
    {
        return -1;
    }
    else
    {
        printf("Begin Insert Elem:%d\n",e);
        int i = 0;
        LinkList *temp = malloc(sizeof(LinkList));  //一个暂时变量
        LinkList *rNext = *list;   //其实感觉这里应该是(*list)->next
        LinkList *q;
        temp->data = e;
        while(i < p-1)            //p-1,这里的bug是运算符优先级，找了好久
        {
            rNext = rNext->next;
            i++;
        }
        //以下是尾插法
        if(!IsEmpty(rNext))
        {
            q = rNext->next;
            rNext->next = temp;
            temp->next = q;
            puts("End Insert."); 
            return 1;
        }
        return -1;
    } 
}
```
#### 删除函数
```c
//删除指定位置的元素,失败返回-1，成功返回1
int DeleteList(LinkList **list,int p,ElemType *e)
{
    if((p<0)||(p>LengthList(*list)))
    {
        return -1;
    }
    else 
    {
        puts("Begin Delete Elem:");
        int i = 0;
        LinkList *rNext = (*list)->next;
        LinkList *temp;
        while(i<p-2)                     //注意是这里p-2,我们要找到删除元素的前一个节点,没毛病
        {                                         
            i++;
            rNext = rNext->next;
        }
        if(!IsEmpty(rNext))
        {          
            temp = rNext->next;
            *e = temp->data;
            rNext->next = temp->next;
            free(temp);
            printf("End Delete:%d\n",*e);
            return 1;
        }

    }
}
```
#### 主函数
```c
//主函数
int main(int argc, char const *argv[])
{
    LinkList *list = NULL;
    ElemType e = 9;
    InitList(&list);
    //插入元素
    InsertList(&list,4,e);
    //删除元素
    DeleteList(&list,4,&e);
    //查找元素
    e = GetElem(list,3);
    printf("Find Elem is :%d\n",e);
    Traverse(list);
    return 0;
}
```
#### 结果
![logo](数据结构之链表/2017-03-15_173012.png)

### 说明：
C语言是没有引用类型的，这是在c++中新定义的类型，其实引用和指针并没有太大的区别，所以我们直接用指针也可以，在本文里出现了二级指针是一个难点。
## 附源码
```c
/**
说明：带头节点链表的C语言实现
author : gsscsd
data : 2017-3-13
**/
#include <stdio.h>
#include <stdlib.h>

#define ElemType int
#define MaxSize 100

//定义链表节点
typedef struct Lnode
{
    ElemType data;
    struct Lnode *next;
}LinkList;

//初始化链表
void InitList(LinkList **list);
//遍历链表
void Traverse(LinkList *list);
//查找指定位置元素
ElemType GetElem(LinkList *list,int p);
//在指定位置插入元素,失败返回-1，成功返回1
int InsertList(LinkList **list,int p,ElemType e);
//删除指定位置的元素,失败返回-1，成功返回1
int DeleteList(LinkList **list,int p,ElemType *e);
//链表是否为空，空返回1
int IsEmpty(LinkList *list);
//计算链表长度，返回链表长度
int LengthList(LinkList *list);

//主函数
int main(int argc, char const *argv[])
{
    LinkList *list = NULL;
    ElemType e = 9;
    InitList(&list);
    //插入元素
    InsertList(&list,4,e);
    //删除元素
    DeleteList(&list,4,&e);
    //查找元素
    e = GetElem(list,3);
    printf("Find Elem is :%d\n",e);
    Traverse(list);
    return 0;
}

//初始化链表
void InitList(LinkList **list)
{
    puts("InitList");
    int i = 0,n = 0;
    *list = malloc(sizeof(LinkList));
    LinkList *temp,*rNext;
    // (*list)->next = NULL;
    rNext = *list;
    puts("Input InitList Numbers:");
    scanf("%d",&n);
    for(i = 0; i < n; i++)
    {
        temp = malloc(sizeof(LinkList));
        puts("Input Number:");
        scanf("%d",&(temp->data));   
        //接下来有两种方法，头插，尾差，这里采用尾差
        rNext->next = temp;
        rNext = rNext->next;
    } 
    rNext->next = NULL;
}

//遍历链表
void Traverse(LinkList *list)
{
	puts("Begin Traverse:");
    LinkList *temp = list->next; //临时变量，方便遍历
    while(temp != NULL)
    {
        printf("%d\t",temp->data);
        temp = temp->next;
    }
    putchar('\n');
    puts("Finished Traverse.");

}

//查找指定位置的元素
ElemType GetElem(LinkList *list,int p)
{
    printf("Find %d Elem:\n",p);
    if((p<0)||(p>LengthList(list)))
    {
        return -1;
    }
    else 
    {
        LinkList *rNext = list->next;
        for(int i = 0;i<p-1;i++)
        {
            rNext = rNext->next;
        }
        puts("End Find.");
        return rNext->data;
    }
}

//在指定位置插入元素,失败返回-1，成功返回1
int InsertList(LinkList **list,int p,ElemType e)
{
    if((p<0)||(p>LengthList(*list)+1))
    {
        return -1;
    }
    else
    {
        printf("Begin Insert Elem:%d\n",e);
        int i = 0;
        LinkList *temp = malloc(sizeof(LinkList));  //一个暂时变量
        LinkList *rNext = *list;   //其实感觉这里应该是(*list)->next
        LinkList *q;
        temp->data = e;
        while(i < p-1)            //p-1,这里的bug是运算符优先级，找了好久
        {
            rNext = rNext->next;
            i++;
        }
        //以下是尾插法
        if(!IsEmpty(rNext))
        {
            q = rNext->next;
            rNext->next = temp;
            temp->next = q;
            puts("End Insert."); 
            return 1;
        }
        return -1;
    } 
}

//删除指定位置的元素,失败返回-1，成功返回1
int DeleteList(LinkList **list,int p,ElemType *e)
{
    if((p<0)||(p>LengthList(*list)))
    {
        return -1;
    }
    else 
    {
        puts("Begin Delete Elem:");
        int i = 0;
        LinkList *rNext = (*list)->next;
        LinkList *temp;
        while(i<p-2)                     //注意是这里p-2,我们要找到删除元素的前一个节点,没毛病
        {                                         
            i++;
            rNext = rNext->next;
        }
        if(!IsEmpty(rNext))
        {          
            temp = rNext->next;
            *e = temp->data;
            rNext->next = temp->next;
            free(temp);
            printf("End Delete:%d\n",*e);
            return 1;
        }

    }
}

//链表是否为空，空返回1
int IsEmpty(LinkList *list)
{
    if(list->next == NULL)
    {
        return 1;
    }
    return 0;
}

//计算链表长度，成功返回链表长度
int LengthList(LinkList *list)
{
    LinkList *rNext = list->next;
    int length = 0;
    while(rNext != NULL)
    {
        length++;
        rNext = rNext->next;
    }   
    return length;
}
```
