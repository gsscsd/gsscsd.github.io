# 剑指Offer之二叉树的镜像


### 题目描述：

操作给定的二叉树，将其变换为源二叉树的镜像。
>
## 输入描述:

```C++
二叉树的镜像定义：源二叉树 
    	    8
    	   /  \
    	  6   10
    	 / \  / \
    	5  7 9  11
    	镜像二叉树
    	    8
    	   /  \
    	  10   6
    	 / \  / \
    	11 9 7   5
```

<!--more-->

### 解题思路：

>时间复杂度: $O(n)$, 空间复杂度: $O(1)$.

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

    void Mirror(TreeNode *pRoot) {
        if(pRoot ==  NULL) return;
        
        TreeNode *temp;
        
        temp = pRoot->right;
        pRoot->right = pRoot->left;
        pRoot->left = temp;
        
        Mirror(pRoot->left);
        Mirror(pRoot->right);
        
    }
};
```


