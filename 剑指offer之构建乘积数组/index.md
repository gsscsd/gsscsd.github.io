# 剑指Offer之构建乘积数组


### 题目描述：

> 给定一个数组A[0,1,...,n-1],请构建一个数组B[0,1,...,n-1],其中B中的元素$$B[i]=A[0]\*A[1]\*...\*A[i-1]\*A[i+1]\*...\*A[n-1]$$。不能使用除法。

<!--more-->

### 解题思路：

> 时间复杂度: $O(n^2)$, 空间复杂度: $O(n)$.

```C++
class Solution {
public:
    vector<int> multiply(const vector<int>& A) {
        vector<int> vec(A.size());
        
        for(int j = 0; j <= A.size()-1; j++)
        {
            int temp  = 1;
            for(int i = 0; i <= A.size()-1; i++)
            {
                if(i != j)
                     temp = A[i] * temp;   
            }
            
            vec[j] = temp;
        }
        
        
        return vec;
    }
};
```

> 时间复杂度: $O(n)$, 空间复杂度: $O(n)$.

```C++
class Solution {
public:
    vector<int> multiply(const vector<int>& A) {
        int n=A.size();
        vector<int> b(n);
        int ret=1;
        for(int i=0;i<n;ret*=A[i++]){     //关键是for语句中的第三条语句，因为它是在循环体之后执行的。
            b[i]=ret;                     //第二是两个for语句中循环体的不同。
        }
        ret=1;
        for(int i=n-1;i>=0;ret*=A[i--]){
            b[i]*=ret;
        }
        return b;
    }
};
```


