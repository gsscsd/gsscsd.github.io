# 剑指Offer之把二叉树打印成多行


### 题目描述：

> 从上到下按层打印二叉树，同一层结点从左至右输出。每一层输出一行。

<!--more-->

### 解题思路：

> 使用队列
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
            vector<vector<int> > vec;
            queue<TreeNode *> q;
            if(!pRoot) return vec;
            q.push(pRoot);
            
            while(!q.empty())
            {
                //这里是重点，要计算队列的长队，好能一行一行的放进去
               int qsize = q.size();
                vector<int> temp;
                for(int i = 0; i < qsize;i++)
                {
                     TreeNode* t = q.front();
                     if(t -> left)q.pu.sh(t->left);
                     if(t -> right)q.push(t->right);
                     temp.push_back(t -> val);
                     q.pop();
                    
                }
                vec.push_back(temp);
            }
            return vec;
        }
    
};
```


