# 数据结构之队列

## <center >前言
>队列是一种特殊的线性表，特殊之处在于它只允许在表的前端（front）进行删除操作，而在表的后端（rear）进行插入操作，和栈一样，队列是一种操作受限制的线性表。进行插入操作的端称为队尾，进行删除操作的端称为队头。队列中没有元素时，称为空队列。
>队列的数据元素又称为队列元素。在队列中插入一个队列元素称为入队，从队列中删除一个队列元素称为出队。因为队列只允许在一端插入，在另一端删除，所以只有最早进入队列的元素才能最先从队列中删除，故队列又称为先进先出（FIFO—first in first out）线性表。

<!--more-->
## 顺序队列的C语言实现
**show code**
```c
/*
说明：循环队列的C语言实现
*/
#include <stdio.h>
#include <stdlib.h>

#define ElemType int
#define MaxSize 100

//顺序栈的数据结构
typedef struct 
{
    ElemType data[MaxSize];
    int rear;   //指示进队的位置
    int front;   //指示出队的位置
}SqQueue;

//初始化队列
void InitQueue(SqQueue *queue);
//进队
void EnQueue(SqQueue *queue,ElemType e);
//出队
void DeQueue(SqQueue *queue,ElemType *e);
//队是否为空
int IsEmpty(SqQueue queue);
//队是否为满
int IsFull(SqQueue queue);
//遍历元素
void Traverse(SqQueue queue);

int main(int argc, char const *argv[])
{
    SqQueue queue;
    int e = 0;
    //初始化队列
    InitQueue(&queue);
    //进队元素
    EnQueue(&queue,1);
    EnQueue(&queue,2);
    //出队元素
    DeQueue(&queue,&e);
    //遍历元素
    Traverse(queue);

    return 0;
}

//初始化队列
void InitQueue(SqQueue *queue)
{
    queue->front = queue->rear = 0;
}

//进队
void EnQueue(SqQueue *queue,ElemType e)
{
    printf("Begin EnQueue data :%d\n",e);
    if(IsFull(*queue))
    {
        puts("It is not enough memory.");
    }
    else 
    {
        queue->rear = (queue->rear + 1)%MaxSize;
        queue->data[queue->rear] = e;
        puts("End EnQueue.");
    }
}

//出队
void DeQueue(SqQueue *queue,ElemType *e)
{
    puts("Begin DeQueue data:");
    if(IsEmpty(*queue))
    {
        puts("It is not data");
    }
    else
    {
        queue->front = (queue->front + 1)%MaxSize;
        *e = queue->data[queue->front];
        printf("End DeQueue .Data is :%d\n",*e);
    }
}
//队是否为空
int IsEmpty(SqQueue queue)
{
    if(queue.front == queue.rear)
    {
        return 1;
    }
    else 
    {
        return 0;
    }
}

//队是否为满
int IsFull(SqQueue queue)
{
    if(queue.front == (queue.rear+1)%MaxSize)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

//遍历元素
void Traverse(SqQueue queue)
{
    puts("Begin Traverse data.");
    if(IsEmpty(queue))
    {
        puts("There is no data.");
    }
    else
    {
        int i = queue.front;
        while(i != queue.rear)
        {
            i = (i+1)%MaxSize;
            printf("%d\t",queue.data[i]);
        }
        putchar('\n');
    }
}
```

## 链队列的C语言实现
**基本show code**
```c
/*
说明：链队的C语言实现
*/
#include <stdio.h>
#include <stdlib.h>

#define ElemType int
#define MaxSize 100

//链队的数据结构
//首先是队的节点结构
typedef struct QNode
{
    ElemType data;
    struct QNode *next;
}QNode;
//定义类型
typedef struct
{
    QNode *front;
    QNode *rear;
}LinkQueue;

//初始化链队
void InitQueue(LinkQueue **queue);
//进队
void EnQueue(LinkQueue **queue,ElemType e);
//出队
void DeQueue(LinkQueue **queue,ElemType *e);
//队是否为空
int IsEmpty(LinkQueue *queue);
//遍历元素
void Traverse(LinkQueue *queue);

int main(int argc, char const *argv[])
{
    LinkQueue *queue;
    int e = 0;
    //初始化链队
    InitQueue(&queue);
    //进队
    EnQueue(&queue,1);
    EnQueue(&queue,2);
    //出队
    DeQueue(&queue,&e);
    //遍历元素
    Traverse(queue);
    return 0;
}

//初始化链队
void InitQueue(LinkQueue **queue)
{
    puts("Begin InitQueue.");
    *queue = malloc(sizeof(LinkQueue));
    (*queue)->front = (*queue)->rear = NULL;
    puts("End InitQueue.");
}

//遍历元素
void Traverse(LinkQueue *queue)
{
    puts("Begin Traverse.");
    if(IsEmpty(queue))
    {
       puts("It is a NULL Queue.");
    }
    else 
    {        
        QNode *q = queue->front; 
        while(q&&q <= queue->rear)  //注意这里容易出Bug
        {
            printf("%d\t",q->data);
            q = q->next;
        }
        putchar('\n');
        puts("End Traverse.");
    }
}

//队是否为空
int IsEmpty(LinkQueue *queue)
{
    //注意这里的判空条件
    if(queue->front == NULL || queue->rear == NULL)
    {
        return 1;
    }
    else 
    {
        return 0;
    }
}

//进队
void EnQueue(LinkQueue **queue,ElemType e)
{
    printf("Begin DeQueue Data is %d\n",e);
    QNode *q = malloc(sizeof(QNode));
    q->data = e;
    q->next = NULL;
    if(IsEmpty(*queue)) ///如果队列为空，则这是一个队头和队尾
    {
        (*queue)->front = (*queue)->rear = q;
    }
    else          //队不空。，直接插到队尾
    {
        (*queue)->rear->next = q;
        (*queue)->rear = q;     //队尾指向q
    }
    puts("End EnQueue.");
}

//出队
void DeQueue(LinkQueue **queue,ElemType *e)
{
    puts("Begin DeQueue.");

    if(IsEmpty(*queue))
    {
        puts("It is a NULL Queue.");
    }
    else 
    {
        QNode *q = malloc(sizeof(QNode));
        q = (*queue)->front;
        if((*queue)->front == (*queue)->rear)//说明队列里只有一个节点
        {
            (*queue)->front = (*queue)->rear = NULL;
        }
        else
        {
            *e = q->data;
            (*queue)->front = q->next;
            free(q);
            printf("End DeQueue Data is :%d\n",*e);
        }
    }
    
}
```
