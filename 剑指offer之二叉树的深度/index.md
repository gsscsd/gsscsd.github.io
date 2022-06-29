# 剑指Offer之二叉树的深度


### 题目描述

> 输入一棵二叉树，求该树的深度。从根结点到叶结点依次经过的结点（含根、叶结点）形成树的一条路径，最长路径的长度为树的深度。

<!--more-->

### 解题思路

> 时间复杂度：${O(n)}$,空间复杂度：$O(1)$.

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

class Solution{
    public:
    int TreeDepth(TreeNode* pRoot)
    {
       int count_left = 0;
       int count_right = 0;
        
       if(pRoot == NULL) return 0;
       if(pRoot->left != NULL) count_left += TreeDepth(pRoot->left);
       if(pRoot->right != NULL) count_right += TreeDepth(pRoot->right);
        
       return max(count_left, count_right)+1;
    }
};

class Solution {
public:
    int TreeDepth(TreeNode* pRoot)
    {
       if(pRoot == NULL) return 0;
       return max(TreeDepth(pRoot->left), TreeDepth(pRoot->right))+1;
    }
};
```


