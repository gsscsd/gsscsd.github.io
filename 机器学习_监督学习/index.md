# 监督学习


### 监督学习：

> 监督学习任务：回归 (用于预测某个值)  和 分类 (用于预测某个分类)
>
> 常见模型：K邻近值算法、线性回归、逻辑回归、支持向量机(SVM)、决策树和随机森林、神经网络

#### 回归：

> 线性回归：线性模型更一般化的描述指通过计算输入变量的加权和，并加上一个常数偏置项（截距项）来得到一个预测值。
>
> 逻辑回归：

##### 线性回归：

>

```python
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)
lin_reg.predict(X_new)
```



#### 分类：

> 二分类：SVM、线性分类
>
> 多分类：随机森林、朴素贝叶斯

##### 梯度下降（GD）：

> 梯度下降的整体思路是通过的迭代来逐渐调整参数使得损失函数达到最小值。
>
> 假设浓雾下，你迷失在了大山中，你只能感受到自己脚下的坡度。为了最快到达山底，一个最好的方法就是沿着坡度最陡的地方下山。这其实就是梯度下降所做的：它计算误差函数关于参数向量                 的局部梯度，同时它沿着梯度下降的方向进行下一次迭代。当梯度值为零的时候，就达到了误差函数最小值 。
>
> 具体来说，开始时，需要选定一个随机的![\theta](https://hand2st.apachecn.org/images/tex-2554a2bb846cffd697389e5dc8912759.gif)（这个值称为随机初始值），然后逐渐去改进它，每一次变化一小步，每一步都试着降低损失函数（例如：均方差损失函数），直到算法收敛到一个最小值。
>
> 在梯度下降中一个重要的参数是步长，超参数学习率的值决定了步长的大小。如果学习率太小，必须经过多次迭代，算法才能收敛，这是非常耗时的。
>
> 另一方面，如果学习率太大，你将跳过最低点，到达山谷的另一面，可能下一次的值比上一次还要大。这可能使的算法是发散的，函数值变得越来越大，永远不可能找到一个好的答案。
>
> 常见模型: 批量梯度下降（Batch GD）、小批量梯度下降（Mini-batch GD）、随机梯度下降（Stochastic GD）

###### 批量梯度下降(Batch GD):

> 批量梯度下降的最要问题是计算每一步的梯度时都需要使用整个训练集，这导致在规模较大的数据集上，其会变得非常的慢。与其完全相反的随机梯度下降，在每一步的梯度计算上只随机选取训练集中的一个样本。很明显，由于每一次的操作都使用了非常少的数据，这样使得算法变得非常快。由于每一次迭代，只需要在内存中有一个实例，这使随机梯度算法可以在大规模训练集上使用。

###### 随机梯度下降分类器(SGD)：

> 这个分类器有一个好处是能够高效地处理非常大的数据集。这部分原因在于SGD一次只处理一条数据，这也使得 SGD 适合在线学习（online learning）。

```python
from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)
```

###### 小批量梯度下降(Mini-batch GD):

> 在迭代的每一步，批量梯度使用整个训练集，随机梯度时候用仅仅一个实例，在小批量梯度下降中，它则使用一个随机的小型实例集。它比随机梯度的主要优点在于你可以通过矩阵运算的硬件优化得到一个较好的训练表现，尤其当你使用 GPU 进行运算的时候。

##### 支持向量机(SVM)：

> 支持向量机（SVM）是个非常强大并且有多种功能的机器学习模型，能够做线性或者非线性的分类，回归，甚至异常值检测。
>
> SVM 特别适合应用于复杂但中小规模数据集的分类问题。

###### 线性支持向量机：

> 以下的 Scikit-Learn 代码加载了内置的鸢尾花（Iris）数据集，缩放特征，并训练一个线性 SVM 模型(使用LinearSVM类，超参数 C = 1，hinge 损失函数)来检测 Virginica 鸢尾花。

```python
import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

iris = datasets.load_iris()
X = iris["data"][:, (2, 3)] # petal length, petal width
y = (iris["target"] == 2).astype(np.float64) # Iris-Virginica

svm_clf = Pipeline((
        ("scaler", StandardScaler()),
        ("linear_svc", LinearSVC(C=1, loss="hinge")),
    ))

svm_clf.fit(X, y)

Then, as usual, you can use the model to make predictions:

 svm_clf.predict([[5.5, 1.7]])
array([ 1.])
```


