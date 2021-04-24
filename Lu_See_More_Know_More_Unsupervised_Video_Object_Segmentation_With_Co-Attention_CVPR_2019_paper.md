基础的 协同注意力
![image](https://user-images.githubusercontent.com/11287531/115958848-5914a980-a55d-11eb-915a-fb47b7e2c066.png)
![image](https://user-images.githubusercontent.com/11287531/115958861-65990200-a55d-11eb-94d4-a5f626cb35e7.png)
![image](https://user-images.githubusercontent.com/11287531/115958874-721d5a80-a55d-11eb-908e-4c64f81f8a27.png)


对称的 协同注意力

如果我们进一步约束权重矩阵W为一个对称矩阵，特征向量组矩阵P就变成了一个正交矩阵。这个对称的协同注意力可以演变成
![image](https://user-images.githubusercontent.com/11287531/115957756-56fc1c00-a558-11eb-8ad1-01fe913367f2.png)
这个公式表明
...？

这个属性已经被证明 对排除不同通道间的相关性 非常有帮助【50】同时有助于提升网络泛化能力【3，48】


基于通道的 协同注意力

而且，特征向量组矩阵P 可以被简化成一个单位矩阵，然后权重矩阵W变成一个对角矩阵。这样的话，W可以被进一步对角化成两个对角矩阵D_a和D_b。
因此，公式3可以写成基于通道的 协同注意力
![image](https://user-images.githubusercontent.com/11287531/115958061-db02d380-a559-11eb-964c-94e32b818c60.png)

这个操作等同于在计算相似性前，先对V_b V_a应用了基于通道的权重。
这对于减少通道相关的冗余很有帮助。和另外2个论文有点像【7，20】。


在ablation study过程中，我们提供了详细的实验细节来显示不同协同注意力机制的效果。
在得到相似度矩阵S之后，我们用softmax分别对S进行行方向的和列方向的归一化。
![image](https://user-images.githubusercontent.com/11287531/115958250-c70ba180-a55a-11eb-91f1-e9c1b9296004.png)

随后，特征V_a关于特征V_b的注意力的summary可以描述为
![image](https://user-images.githubusercontent.com/11287531/115958354-3b464500-a55b-11eb-8299-71a9d894dd37.png)
相似地，也可以计算特征V_b关于特征V_a的注意力的summary。

考虑到输入的两帧的潜在外观变化，遮挡还有背景噪音，我们应该给不同帧不同权重，而不是平等的对待所有协同注意力信息。
为此，引入了一种self-gate机制，可以为每个注意力摘要分配 协同注意力置信度。
这个gate的公式如下
![image](https://user-images.githubusercontent.com/11287531/115958598-36ce5c00-a55c-11eb-8838-fc670c65919c.png)
σ 是个逻辑sigmoid激活函数，w_f和b_f是卷积权重和bias。
f_g确定了有多少来自于引用帧的信息被保留，可以被自动学习。
计算了gate置信度后，注意力的摘要可以描述成
![image](https://user-images.githubusercontent.com/11287531/115958732-b1977700-a55c-11eb-902d-e6714858ebc2.png)

然后，把这个协同注意力Z和原始的特征V连接起来
![image](https://user-images.githubusercontent.com/11287531/115958770-e3a8d900-a55c-11eb-9c94-7947dc884111.png)

最后，被协同注意力强化了的特征X被送往一个分割网络用来产生最终的mask。
