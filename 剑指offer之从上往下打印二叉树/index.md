# 剑指Offer之从上往下打印二叉树


### 题目描述：

> 从上往下打印出二叉树的每个节点，同层节点从左至右打印。

<!--more-->

### 解题思路：

> 二叉树的层次遍历么，借助一个队列。
>
> 时间复杂度：$O(n)$, 空间复杂度：$O(n)$.

```C++
/*
struct TreeNode {
	int val;
	struct TreeNode *left;
	struct TreeNode *right;
	TreeNode(int x) :
			val(x), left(NULL), right(NULL) {
	}
};*/
class Solution {
public:
    vector<int> PrintFromTopToBottom(TreeNode* root) {
        vector<int> vec;
        queue<TreeNode*> q;
        
        if(!root) return vec;
        q.push(root);
        
        while(!q.empty())
        {
            vec.push_back(q.front()->val);
            if(q.front()->left)  q.push(q.front()->left);
            if(q.front()->right) q.push(q.front()->right);
            q.pop();
        }
        
        return vec;
    }
};
```


