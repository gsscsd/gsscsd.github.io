# 剑指Offer之按之字形顺序打印二叉树


### 题目描述：

> 请实现一个函数按照之字形打印二叉树，即第一行按照从左到右的顺序打印，第二层按照从右至左的顺序打印，第三行按照从左到右的顺序打印，其他行以此类推。

<!--more-->

### 解题思路：

> 使用两个栈进行转化
>
> 时间复杂度: $O(n)$, 空间复杂度: $O(n)$.

```C++
/*
struct TreeNode {
    int val;
    struct TreeNode *left;
    struct TreeNode *right;
    TreeNode(int x) :
            val(x), left(NULL), right(NULL) {
    }
};
*/
class Solution {
public:
    vector<vector<int> > Print(TreeNode* pRoot) {
        // 奇数层
        stack<TreeNode *> s1;
        // 偶数层
        stack<TreeNode *> s2;
        vector<vector<int> > vec;
        
        if(!pRoot) return vec;
        s1.push(pRoot);
        // 记录放在了那个栈里面
        int flag = 0;
                
        while(!s1.empty() || !s2.empty())
        {
            int ssize1 = s1.size();
            vector<int> temp;
            
            for(int i = 0; i < ssize1;i++)
            {
                 TreeNode* t1 = s1.top();
                 if(t1 -> left)s2.push(t1->left);
                 if(t1 -> right)s2.push(t1->right);
                 temp.push_back(t1 -> val);
                 s1.pop();
                 flag = 1;
             }
            // 只有走s1时才放入到vec中
            if(flag == 1)
            {
                 vec.push_back(temp);
                 temp.clear();
            }
            int ssize2 = s2.size();
            for(int j = 0; j < ssize2;j++)
            {
                 TreeNode* t2 = s2.top();
                 if(t2 -> right)s1.push(t2->right);
                 if(t2 -> left)s1.push(t2->left);
                 temp.push_back(t2 -> val);
                 s2.pop();
                 flag = 2;
            }
             // 只有走s2时才放入到vec中
            if(flag == 2)
            {
                 vec.push_back(temp);
                 temp.clear();
            }
        }
        return vec;
    }
};
```


