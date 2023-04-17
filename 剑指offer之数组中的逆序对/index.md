# 剑指Offer之数组中的逆序对


### 题目描述：

在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。输入一个数组,求出这个数组中的逆序对的总数P。并将P对1000000007取模的结果输出。 即输出P%1000000007
#### 输入描述:

题目保证输入的数组中没有的相同的数字数据范围：	对于%50的数据,size<=10^4对于%75的数据,size<=10^5	对于%100的数据,size<=2*10^5

示例1

#### 输入

```C++
1,2,3,4,5,6,7,0
```

#### 输出

```C++
 7
```

<!--more-->

### 解题思路：

>先把数组分割成子数组，先统计出子数组内部的逆序对的数目，然后再统计出两个相邻子数组之间的逆序对的数目。在统计逆序对的过程中，还需要对数组进行排序。
时间复杂度：$$O(nlogn)$$,空间复杂度：$$O(n)$$.

```C++
class Solution {
public:
    long countRes ;
    int InversePairs(vector<int> data) {
        countRes = 0;
        if(data.size() == 0)
            return 0;
        // 调用归并排序
        MergeSort(data,0,data.size()-1);
        return countRes%1000000007 ;
    }
    // 归并排序
    void MergeSort(vector<int>& data,int first,int end){
        if(first < end){
            int mid = (first + end)/2;
            MergeSort(data,first,mid);
            MergeSort(data,mid+1,end);
            vector<int> tmp;
            MergeArray(data,first,mid,end,tmp);
        }
    }
    void MergeArray(vector<int>& data,int first,int mid,int end,vector<int> tmp){
        int i = first;int m = mid;
        int j = mid + 1;int n = end;
     	
        while(i<=m && j<=n){
            if(data[i] > data[j]){
                tmp.push_back(data[i++]);
                countRes += n - j + 1; // *****
            }
            else{
                tmp.push_back(data[j++]);
            }
        }
        while(i<=m)
            tmp.push_back(data[i++]);
        while (j<=n)
            tmp.push_back(data[j++]);
 
        //更新data数组
        int k = 0;
        for (int i = first; i <= end &&k<tmp.size(); i++)
            data[i] = tmp[k++];
         
    }
};

```


