# 剑指Offer之重建二叉树


### 题目描述:

> 输入某二叉树的前序遍历和中序遍历的结果，请重建出该二叉树。假设输入的前序遍历和中序遍历的结果中都不含重复的数字。例如输入前序遍历序列{1,2,4,7,3,5,6,8}和中序遍历序列{4,7,2,1,5,3,8,6}，则重建二叉树并返回。

<!--more-->

### 解题思路一：

> 时间复杂度：$O(n)$, 空间复杂度：$O(n)$.

```C++
/**
 * Definition for binary tree
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    TreeNode* reConstructBinaryTree(vector<int> pre,vector<int> vin) {

            int inlen=vin.size();
            if(inlen==0) 
                return NULL;
 
            vector<int> left_pre,right_pre,left_in,right_in;
 
            //创建根节点，根节点肯定是前序遍历的第一个数
            TreeNode* head=new TreeNode(pre[0]);

            //找到中序遍历根节点所在位置,存放于变量gen中
            int gen=0;
            for(int i=0;i<inlen;i++)
            {
                if (vin[i]==pre[0])
                {
                    gen=i;
                    break;
                }
            }
            //对于中序遍历，根节点左边的节点位于二叉树的左边，根节点右边的节点位于二叉树的右边
            //利用上述这点，对二叉树节点进行归并
            for(int i=0;i<gen;i++)
            {
                left_in.push_back(vin[i]);
                left_pre.push_back(pre[i+1]);//前序第一个为根节点
            }
            for(int i=gen+1;i<inlen;i++)
            {
                right_in.push_back(vin[i]);
                right_pre.push_back(pre[i]);
            }
            //和shell排序的思想类似，取出前序和中序遍历根节点左边和右边的子树
            //递归，再对其进行上述所有步骤，即再区分子树的左、右子子数，直到叶节点
           head->left=reConstructBinaryTree(left_pre,left_in);
           head->right=reConstructBinaryTree(right_pre,right_in);
 
           return head;
        }
    };
```

### 解题思路二：

> 时间复杂度：$O(n)$, 空间复杂度：$O(1)$.

```C++
/**
 * Definition for binary tree
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    TreeNode* reConstructBinaryTree(vector<int> pre,vector<int> vin) {
        TreeNode *root = isreConstructBinaryTree(pre,0,pre.size()-1,vin,0,vin.size()-1);
        return root;
    }
    
    TreeNode* isreConstructBinaryTree(vector<int> pre,int startPre,int endPre,vector<int> in,int startIn,int endIn) {
         
        if(startPre>endPre||startIn>endIn)
            return NULL;
        TreeNode *root=new TreeNode(pre[startPre]);
         
        for(int i = startIn;i <= endIn;i++)
        {
            if(in[i] == pre[startPre])
            {
                root->left = isreConstructBinaryTree(pre,startPre+1,startPre+i-        		   startIn,in,startIn,i-1);
                root->right = isreConstructBinaryTree(pre,i-startIn+startPre+1,endPre,in,i+1,endIn);
                break;
            }
        }
                 
        return root;
    }
};
```


