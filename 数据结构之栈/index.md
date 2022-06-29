# 数据结构之栈

## <center >前言
>栈（stack）又名堆栈，它是一种运算受限的线性表。其限制是仅允许在表的一端进行插入和删除运算。这一端被称为栈顶，相对地，把另一端称为栈底。向一个栈插入新元素又称作进栈、入栈或压栈，它是把新元素放到栈顶元素的上面，使之成为新的栈顶元素；从一个栈删除元素又称作出栈或退栈，它是把栈顶元素删除掉，使其相邻的元素成为新的栈顶元素。
>栈作为一种数据结构，是一种只能在一端进行插入和删除操作的特殊线性表。它按照先进后出的原则存储数据，先进入的数据被压入栈底，最后的数据在栈顶，需要读数据的时候从栈顶开始弹出数据（最后一个数据被第一个读出来）。栈具有记忆作用，对栈的插入与删除操作中，不需要改变栈底指针。

<!--more-->

## 顺序栈的C语言实现
**show code:**
```c
/*
说明：顺序栈的C语言实现
*/

#include <stdio.h>
#include <stdlib.h>

#define ElemType int
#define MaxSize 100

//顺序栈的数据结构
typedef struct 
{
    ElemType data[MaxSize];
    int top;
}SqlStack;

//初始化栈
void InitStack(SqlStack *stack);
//栈顶插入元素
void push(SqlStack *stack,ElemType e);
//栈顶删除元素
void pop(SqlStack *stack,ElemType *e);
//栈是否为空
int IsEmpty(SqlStack stack);
//栈是否为满
int IsFull(SqlStack stack);
//输出栈内元素
void Traverse(SqlStack stack);

//主函数
int main(int argc, char const *argv[])
{
    SqlStack stack;
    int e = 0;
    //初始化栈
    InitStack(&stack);
    //printf("%d",stack.top);
    //进栈
    push(&stack,1);
    push(&stack,2);
    //出栈 
    pop(&stack,&e);
    //遍历元素 
    Traverse(stack);
    
    return 0;
}

//初始化栈
void InitStack(SqlStack *stack)
{
	puts("Begin InitStack:");
    stack->top = -1;  //赋值为-1为了节省一个内存空间，也可为0
	puts("End InitStack.");

}
//栈顶插入元素
void push(SqlStack *stack,ElemType e)
{
    if(IsFull(*stack))
    {
        puts("No Enough Memory.");
    }
    else 
    {
		printf("Begin Push Data:%d\n",e);
        stack->data[++(stack->top)] = e;
		puts("End Push.");
    }
}
//栈顶删除元素
void pop(SqlStack *stack,ElemType *e)
{
	puts("Begin Pop Data:");
    if(IsEmpty(*stack))
    {
        puts("it is a NULL Stack.");
    }
    else 
    {
        *e = stack->data[stack->top--];
		printf("end pop data is :%d\n",*e);
    }

}
//栈是否为空
int IsEmpty(SqlStack stack)
{
    if(stack.top == -1)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}
//输出栈内元素
void Traverse(SqlStack stack)
{
    if(!IsEmpty(stack))
    {
        puts("Traverse:");
        for(int i = stack.top;i>-1;i--)
        {
            printf("%d\t",stack.data[i]);
        }
        putchar('\n');
    }

}

//栈是否为满
int IsFull(SqlStack stack)
{
    if(stack.top == MaxSize - 1)
    {
        return 1;
    }
    else 
    {
        return 0;
    }
}
```
## 链栈的C语言实现
**show cod**
```c
/*
说明:带有头节点链栈的C语言实现
*/
#include <stdio.h>
#include <stdlib.h>

#define ElemType int
#define MaxSize 100

//链栈的数据结构
typedef struct StackNode
{
    ElemType data;
    struct StackNode *next;   
}LinkStack;

//初始化链栈
void InitStack(LinkStack **stack);
//压栈
void push(LinkStack **stack,ElemType e);
//出栈
void pop(LinkStack **stack,ElemType *e);
//遍历栈元素
void Traverse(LinkStack *stack);
//栈是否为空
int IsEmpty(LinkStack *stack);
//获取栈顶元素
ElemType GetElem(LinkStack *stack);

//主函数
int main(int argc, char const *argv[])
{
    LinkStack *stack;
    int e = 0;
    //初始化链栈
    InitStack(&stack);
    //进栈数据
    push(&stack,1);
    push(&stack,2);
    push(&stack,3);
    //获取栈顶元素
    printf("GetElem pop is :%d\n",GetElem(stack));
    //出栈
    pop(&stack,&e);
    //遍历数据
    Traverse(stack);
    return 0;
}

//初始化链栈
void InitStack(LinkStack **stack)
{
    puts("Begin InitStack:");
    *stack = malloc(sizeof(LinkStack));
    (*stack)->next = NULL;
    puts("End InitStack.");
}

//栈是否为空
int IsEmpty(LinkStack *stack)
{
    if(stack->next == NULL)
    {
        return 1;
    }
    else
    {
        return 0;
    }
    
}

//压栈,采用头插法
void push(LinkStack **stack,ElemType e)
{
    printf("Begin Push Data:%d\n",e);
    LinkStack *p = malloc(sizeof(LinkStack));
    p->data = e;
    p->next = (*stack)->next;
    (*stack)->next = p;
    puts("End Push.");
}

//出栈
void pop(LinkStack **stack,ElemType *e)
{
    puts("Begin Pop Data:");
    if(IsEmpty(*stack))
    {
        puts("it is a NUll  Stack");
    }
    else
    {
        LinkStack *temp = NULL;
        temp = (*stack)->next;
        *e = temp->data;
        (*stack)->next = temp->next;
        free(temp);          //释放申请的资源
        printf("end pop data is :%d\n",*e);
    }
}

//遍历栈元素
void Traverse(LinkStack *stack)
{
    puts("Begin Traverse stack:");
    LinkStack *rNext = stack->next;
    while(rNext != NULL)
    {
        printf("%d\t",rNext->data);
        rNext = rNext->next;
    }
    putchar('\n');
    puts("End Traverse Data.");
}

//获取栈顶元素
ElemType GetElem(LinkStack *stack)
{
    puts("Bengin GetElem:");
    if(IsEmpty(stack))
    {
        puts("It is no Data.");
        return 0;
    }
    else 
    {
        return stack->next->data;   
    }
}
```


