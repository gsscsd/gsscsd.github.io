# 剑指Offer之表示数值的字符串


### 题目描述：

> 请实现一个函数用来判断字符串是否表示数值（包括整数和小数）。例如，字符串"+100","5e2","-123","3.1416"和"-1E-16"都表示数值。 但是"12e","1a3.14","1.2.3","+-5"和"12e+4.3"都不是。

<!--more-->

### 解题思路一：

> 几个关键点： 
>
> 1.基本边界     string == NULL || *string == '\0' 
>
> 2.检测是否有符号位(检测第一位) 
>
> 3.检测除符号位外的第一个有效位，有效位只能是数字或者小数点.   
>
> 4.检测是否有E或者e，且不能重复出现
>
> 5.小数点不能重复出现 ，E或者e后面不能出现小数点
>
> 6.中间的符号位必须出现在E或者e后面
>
> 7.数字的合法性，不能是其他字母如‘a’等
>
> 设置参数hasPoint、hasEe来判断E、e、小数点是否重复出现
>
> 时间复杂度：$O(n), $空间复杂度：$O(1)$.

```C++
class Solution {
public:
bool isNumeric(char* string)
    {
        bool hasPoint = false;
        bool hasEe = false;
        
        // 基本边界
        if(string == NULL || *string == '\0') return false;
        // 检测是否有符号位
        bool isMinus = false;
        if(*string == '-') 
        {
            isMinus = true;
            string++;
        } 
        else if(*string == '+')
        {
            isMinus = false;
            string++;
        }
 
        // 检测第一个数字或者小数点是否存在
        if((*string >= '0' && *string <= '9')) string++;
        if(*string == '.')
        {
            hasPoint = true;
            string++;
        }
        while(*string != '\0') 
        {
            // 是否为E或者e
            if(*string == 'E' || *string == 'e') 
            {
                if(hasEe == true) 
                {
                    return false;
                } 
                else 
                {
                    hasEe = true;
                    string++;
                }
            }
            // 是否为小数点.
            if(*string == '.') 
            {
                if(hasPoint == true) 
                {
                    return false;
                } 
                else 
                {
                    if(hasEe == true) return false;
                    hasPoint = true; 
                    string++;
                }
            } 
            // 是否为符号
            if(*string == '-' || *string == '+') 
            {
                if(hasEe == true) string++;
                else return false;
            }
            // 是否为合法数字
            else if(*string >= '0' && *string <= '9') string++;
            else return false;
        }
 
        // 如果不是所有不合法，则返回true
        return true;
    }
};
```

### 解题思路二：

> 注意表示数值的字符串遵循的规则；
> 在数值之前可能有一个“+”或“-”，接下来是0到9的数位表示数值的整数部分，如果数值是一个小数，那么小数点后面可能会有若干个0到9的数位
> 表示数值的小数部分。如果用科学计数法表示，接下来是一个‘e’或者‘E’，以及紧跟着一个整数（可以有正负号）表示指数。
>
> 时间复杂度：$O(n)$,空间复杂度：$O(1)$.

```C++
 bool isNumeric(char* string)
    {
        if(string==NULL)
            return false;
        if(*string=='+'||*string=='-')
            string++;
        if(*string=='\0')
            return false;
        int dot=0,num=0,nume=0;//分别用来标记小数点、整数部分和e指数是否存在
        while(*string!='\0'){
            if(*string>='0'&&*string<='9')
            {  
                string++;
                num=1;
            }
            else if(*string=='.'){
                if(dot>0||nume>0)
                    return false;
                string++;
                dot=1;
            }
            else if(*string=='e'||*string=='E')
                {
                  if(num==0||nume>0)
                      return false;
                  string++;
                  nume++;
                  if(*string=='+'||*string=='-')
                      string++;
                 if(*string=='\0')
                     return false;
                }
            else
                return false;
        }
        return true;
```


