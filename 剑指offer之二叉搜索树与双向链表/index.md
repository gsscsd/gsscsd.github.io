# 剑指Offer之二叉搜索树与双向链表


### 题目描述：

> 输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的双向链表。要求不能创建任何新的结点，只能调整树中结点指针的指向。

<!--more-->

### 解题思路一：

> 1 .将左子树构造成双链表，并返回链表头节点。
> 2 .定位至左子树双链表最后一个节点。
> 3 .如果左子树链表不为空的话，将当前root追加到左子树链表。
> 4 .将右子树构造成双链表，并返回链表头节点。
> 5 .如果右子树链表不为空的话，将该链表追加到root节点之后。
> 6 .根据左子树链表是否为空确定返回的节点。
>
> 时间复杂度：$O(n)$, 空间复杂度$O(1)$.

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
    TreeNode* Convert(TreeNode* pRootOfTree)
    {
        //if(!pRootOfTree || !(pRootOfTree->left && pRootOfTree->right)) return pRootOfTree;
        if(!pRootOfTree) return NULL;
        if(!pRootOfTree->left && !pRootOfTree->right) return pRootOfTree;
        
        
        TreeNode *left = NULL;
        TreeNode *p = NULL;
        TreeNode *right = NULL;
       // 1.将左子树构造成双链表，并返回链表头节点 
       left = Convert(pRootOfTree->left);
       p = left;
       // 2.定位至左子树双链表最后一个节点
       while(p && p->right)
       {
           p=p->right;
       }
       // 3.如果左子树链表不为空的话，将当前root追加到左子树链表 
       if(left)
       {
           p->right = pRootOfTree;
           pRootOfTree->left = p;
       }
        // 4.将右子树构造成双链表，并返回链表头节点
        right = Convert(pRootOfTree->right);
        // 5.如果右子树链表不为空的话，将该链表追加到root节点之后
        if(right)
        {
            right->left = pRootOfTree;
            pRootOfTree->right = right;
        }
        return left != NULL? left:pRootOfTree; 
    }
};
```

### 解题思路二：

> 思路与方法一中的递归版一致，仅对第2点中的定位作了修改，新增一个全局变量记录左子树的最后一个节点。
>
> 时间复杂度：$O(n)$, 空间复杂度$O(1)$.

```C++
   // 记录子树链表的最后一个节点，终结点只可能为只含左子树的非叶节点与叶节点
class Solution {
protected:
    TreeNode *leftLast = NULL;
public:
    TreeNode* Convert(TreeNode* pRootOfTree)
    {
        if(pRootOfTree==NULL)
            return NULL;
        if(pRootOfTree->left==NULL&&pRootOfTree->right==NULL){
            leftLast = pRootOfTree;// 最后的一个节点可能为最右侧的叶节点
            return pRootOfTree;
        }
        // 1.将左子树构造成双链表，并返回链表头节点
        TreeNode *left = Convert(pRootOfTree->left);
        // 3.如果左子树链表不为空的话，将当前root追加到左子树链表
        if(left!=null){
            leftLast->right = pRootOfTree;
            pRootOfTree->left = leftLast;
        }
        leftLast = pRootOfTree;// 当根节点只含左子树时，则该根节点为最后一个节点
        // 4.将右子树构造成双链表，并返回链表头节点
        TreeNode *right = Convert(pRootOfTree->right);
        // 5.如果右子树链表不为空的话，将该链表追加到root节点之后
        if(right!=null){
            right->left = pRootOfTree;
            pRootOfTree->right = right;
        }
        return left!=null?left:pRootOfTree;       
    }

```


