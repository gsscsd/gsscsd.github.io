# 剑指Offer之机器人的运动范围


### 题目描述：

> 地上有一个m行和n列的方格。一个机器人从坐标0,0的格子开始移动，每一次只能向左，右，上，下四个方向移动一格，但是不能进入行坐标和列坐标的数位之和大于k的格子。 例如，当k为18时，机器人能够进入方格（35,37），因为3+5+3+7 = 18。但是，它不能进入方格（35,38），因为3+5+3+8 = 19。请问该机器人能够达到多少个格子？

<!--more-->

### 解题思路：

>核心思路：
>
>1.从(0,0)开始走，每成功走一步标记当前位置为true,然后从当前位置往四个方向探索，
>
>返回1 + 4 个方向的探索值之和。
>
>2.探索时，判断当前节点是否可达的标准为：
>
>1）当前节点在矩阵内；
>
>2）当前节点未被访问过；
>
>3）当前节点满足limit限制。
>
>// 注意他的分析思路
>
>比如说，节点是否可以访问，用函数来处理，位置数值的分解用函数来处理

```C++
class Solution {
public:
    int movingCount(int threshold, int rows, int cols)
    {
        int count = 0;
        bool *flag = new bool[rows * cols]();
        count = move(threshold,rows,cols,0,0,flag);
        return count;
        
    }
    // 判断是否能移动
    bool isMove(int rows,int cols,int i,int j,int threshold,bool *flag)
    {
        if(i >= 0 && j >= 0&& i < rows && j < cols && !flag[i * cols + j] && getNum(i) + getNum(j) <= threshold) return true;
        return false;
    }
    int getNum(int k)
    {
        int res = 0;
        while(k > 0 )
        {
            res += k % 10;
            k /= 10;
        }
        return res;
    }
    // 递归的移动函数
    int move(int threshold,int rows,int cols,int i,int j,bool *flag)
    {
        int count = 0;
        if(isMove(rows,cols,i,j,threshold,flag))
        {
            flag[i * cols + j] = true;
            count = 1 + move(threshold,rows,cols,i-1,j,flag)
                + move(threshold,rows,cols,i+1,j,flag)
                + move(threshold,rows,cols,i,j - 1,flag)
                + move(threshold,rows,cols,i,j + 1,flag);
        }
        return count;
    }
};
```


