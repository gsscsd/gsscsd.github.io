# 剑指Offer之二叉树中和为某一值的路径


### 题目描述：

> 输入一颗二叉树的跟节点和一个整数，打印出二叉树中结点值的和为输入整数的所有路径。路径定义为从树的根结点开始往下一直到叶结点所经过的结点形成一条路径。(注意: 在返回值的list中，数组长度大的数组靠前)

<!--more-->

### 解题思路：

> 时间复杂度：$O(n)$, 空间复杂度$O(n)$.

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
    // 定义全局变量
    vector<vector<int> >  vec;
    vector<int> temp;
    vector<vector<int> > FindPath(TreeNode* root,int expectNumber) {

        // 如果root为NULL
        if(!root) return vec;
        // 将root的数据放入到temp里面
        temp.push_back(root -> val);
        // target减去root的值
        expectNumber -= root -> val; 
        // 如果减去的数为0并且是叶子节点，那么将temp放入发到vec
        if(expectNumber == 0 && root->left == NULL && root->right == NULL)
        {
            vec.push_back(temp);
        }
        // 递归计算左子树
        FindPath(root->left,expectNumber);
        // 递归计算右子树
        FindPath(root->right,expectNumber);
        // 回溯，弹出本层的数据
        temp.pop_back();
        return vec;
    }
};
```


