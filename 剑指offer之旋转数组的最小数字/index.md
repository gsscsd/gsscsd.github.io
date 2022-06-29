# 剑指Offer之旋转数组的最小数字


### 题目描述：

> 把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。 输入一个非减排序的数组的一个旋转，输出旋转数组的最小元素。 例如数组{3,4,5,1,2}为{1,2,3,4,5}的一个旋转，该数组的最小值为1。 NOTE：给出的所有元素都大于0，若数组大小为0，请返回0。

<!--more-->

### 解题思路一：

> 时间复杂度: $O(n)$, 空间复杂度: $O(1)$.

```C++
class Solution {
public:
    int minNumberInRotateArray(vector<int> rotateArray) {
        if(rotateArray.size() == 0) return 0;
        
         int temp = rotateArray[0];
        
        for(int i = 0; i < rotateArray.size(); i++)
        {
            if(temp > rotateArray[i])
                temp = rotateArray[i];
        }
        
        return temp;
    }
};
```

### 解题思路二：

> 时间复杂度: $O(n)$, 空间复杂度: $O(1)$.

```C++
class Solution {
public:
    int minNumberInRotateArray(vector<int> rotateArray) {
        
       if (rotateArray.size() == 0) return 0;
       int temp = rotateArray[0];
       int i = 0;
       int j = rotateArray.size() - 1;
        
       while(i < j)
       {
           if(rotateArray[i] <= rotateArray[j])
           {
               temp = rotateArray[i];
               j--;
           }
           else
           {
               temp = rotateArray[j];
               i++;
           }
       }
        
        return temp;
    }
};
```

### 解题思路三：

> 采用二分法解答这个问题，
>
> mid = low + (high - low)/2
>
> 需要考虑三种情况：
>
> (1)array[mid] > array[high]:
>
> 出现这种情况的array类似[3,4,5,6,0,1,2]，此时最小数字一定在mid的右边。
>
> low = mid + 1
>
> (2)array[mid] == array[high]:
>
> 出现这种情况的array类似 [1,0,1,1,1] 或者[1,1,1,0,1]，此时最小数字不好判断在mid左边
>
> 还是右边,这时只好一个一个试 ，
>
> high = high - 1
>
> (3)array[mid] < array[high]:
>
> 出现这种情况的array类似[2,2,3,4,5,6,6],此时最小数字一定就是array[mid]或者在mid的左
>
> 边。因为右边必然都是递增的。
>
> high = mid
>
> 时间复杂度: $O(logn)$, 空间复杂度: $O(1)$.

```C++
class Solution {
public:
    int minNumberInRotateArray(vector<int> rotateArray) {
        
        int low = 0 ; 
        int high = rotateArray.size() - 1;   
        while(low < high){
            int mid = low + (high - low) / 2;        
            if(rotateArray[mid] > rotateArray[high]){
                low = mid + 1;
            }else if(rotateArray[mid] == rotateArray[high]){
                high = high - 1;
            }else{
                high = mid;
            }   
        }
        return rotateArray[low];
    }
};
```


