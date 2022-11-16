> 参考博客：
> 
> &emsp; [理解学习率和 batch_size 的关系](https://zhuanlan.zhihu.com/p/364865720)
> 
> &emsp; [理解学习率和 batch_size 的关系](https://zhuanlan.zhihu.com/p/364865720)
> 
> 参考论文：
> 
> &emsp; [Accurate Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677)

> **问题：根据线性放缩原则(linear scaling rule), 学习率的设置应该和 batchsize 成正比，但是为什么有这样的关系？**

### 线性放缩原则
$$
w_{t+1} = w_t - \eta  \frac{1}{n} \sum_{x \in B} \nabla l(x, w_t)
$$

其中: $\eta$ 为学习率，$n$ 为 Batchsize 大小，$\nabla l(x, w_t)$ 为一次训练所得的 loss 值的梯度。<br>
如果可以假设 $\nabla l(x, w_t)$ 在任意 t 都近似相等，则 $\eta$ 与 $n$ 即为等比关系。

**当然，这样的假设太过牵强，有人提出相关其他假设，或者从梯度方差考虑(这种方法推荐 batchsize 和 lr 的放缩关系为 $k : \sqrt{k}$)。**

**由于相关内容过多，如果想深入了解，推荐看参考论文和相关博客。**