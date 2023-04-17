# 剑指Offer之平衡二叉树


### 题目描述：

> 输入一棵二叉树，判断该二叉树是否是平衡二叉树。

<!--more-->

### 解题思路:

> 时间复杂度: $O(n^2)$,空间复杂度：$O(1)$.

```C++
class Solution {
public:
    // 递归的判断是否是平衡二叉树
    bool IsBalanced_Solution(TreeNode* pRoot) {
        // 如果初始节点是NULL，直接返回true
        if(pRoot == NULL ) return true;
        int  left = 0, right = 0;
        
        // 递归计算左子树的深度
        left = dep_tree(pRoot->left);
        // 递归的计算右子树的深度
        right = dep_tree(pRoot->right);
		// 判断两个深度的差值是否大于1，如果大于1，返回false
        if(abs(left - right) > 1)
            return false;
        // 否则，递归的计算左子树和右子树
        else 
            return IsBalanced_Solution(pRoot -> left) && IsBalanced_Solution(pRoot -> right);
     }
    
    // 计算树的深度
    int dep_tree(TreeNode* pRoot)
    {
        if(!pRoot) return 0;
       
        return max(dep_tree(pRoot->left),dep_tree(pRoot->right))+1;
    }
};
```


