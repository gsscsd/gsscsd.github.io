# 剑指Offer之对称的二叉树


### 题目描述：

> 请实现一个函数，用来判断一颗二叉树是不是对称的。注意，如果一个二叉树同此二叉树的镜像是同样的，定义其为对称的。

<!--more-->

### 解题思路：

> 时间复杂度：$O(n)$, 空间复杂度：$O(1)$.

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
    // 递归的判断是否是对成树
    bool isSymmetrical(TreeNode* pRoot)
    {
        //如果为空，返回true
        if(pRoot == NULL) return true;
        // 递归判断子树
        return isbt(pRoot->left,pRoot->right);
    }
    // 递归判断对称树
    bool  isbt(TreeNode* left, TreeNode* right)
    {
        // 如果左右子树都为NULL，则返回true
        if(!left && !right) return true;
        // 如果左右子树有一个为NULL，返回false
        if(!left || !right) return false;
        // 如果左子树==右子树
        if(left->val == right->val)
            // 递归的判断左子树的左子树与右子树的右子树
            // 以及左子树的右子树和右子树的左子树是否对称
            return isbt(left->left, right->right) &&
                   isbt(left->right, right->left);
        return false;
    }

};
```


