# 剑指Offer之替换空格


### 题目描述：

> 请实现一个函数，将一个字符串中的每个空格替换成“%20”。例如，当字符串为We Are Happy.则经过替换之后的字符串为We%20Are%20Happy。

<!--more-->

### 解题思路：

> 时间复杂度: $O(n)$, 空间复杂度: $O(1)$.

```C++
class Solution {
public:
    void replaceSpace(char *str,int length) {
        int count=0;
        for(int i=0; i<length; i++) {
            if(str[i]==' ') 
            {
                count++;
            }
        }
        
        int new_length = count * 2 + length;
        int j = new_length - 1;
        
        for(; length>0; length--)
            if(str[length-1]==' ') 
            {
                str[j--]='0';
                str[j--]='2';
                str[j--]='%';
            } 
            else 
            {
                str[j--]=str[length-1];
            }
    }
};
```


