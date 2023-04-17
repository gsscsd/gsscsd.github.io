# 剑指Offer之树的子结构


### 题目描述：

> 输入两棵二叉树A，B，判断B是不是A的子结构。（ps：我们约定空树不是任意一个树的子结构）

<!--more-->

### 解题思路：

> 时间复杂度: $O(n)$， 空间复杂度: $O(1)$.

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
    bool HasSubtree(TreeNode* pRoot1, TreeNode* pRoot2)
    {
        bool flag = false;
        
        if(pRoot1 && pRoot2)
        {
             //如果找到了对应pRoot2的根节点的点
            if(pRoot1->val == pRoot2->val)
                 //以这个根节点为为起点判断是否包含pRoot2
                flag = isHasSubtree(pRoot1,pRoot2);
            //如果找不到，那么就再去pRoot1的左儿子当作起点，去判断是否包含pRoot2
            if(!flag)  flag = HasSubtree(pRoot1->left,pRoot2);
            //如果找不到，那么就再去pRoot1的右儿子当作起点，去判断是否包含pRoot2
            if(!flag)  flag = HasSubtree(pRoot1->right,pRoot2);
                
        }
        
        return flag;
    } 
     bool isHasSubtree(TreeNode* pRoot1, TreeNode* pRoot2) 
     {
         if(pRoot2 == NULL) return true;
         if(pRoot1 == NULL) return false;
         if(pRoot1->val != pRoot2->val) return false;
         
         return isHasSubtree(pRoot1->left, pRoot2->left) && isHasSubtree(pRoot1->right, pRoot2->right);
     }
};
```


