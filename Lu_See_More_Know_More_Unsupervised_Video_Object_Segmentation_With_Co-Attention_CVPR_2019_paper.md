```diff
- 以下只有绿色字体和斜字体是我的问题，其他字体是paper或其翻译。
```

======Mechanisms============================</br>
**基础的 协同注意力**

Input: 视频里的两帧 F<sub>a</sub>, F<sub>b</sub></br>
提取这两帧的feature V<sub>a</sub>, V<sub>b</sub>。 V<sub>a</sub>∈R<sup><I>W×H×C</I></sup>，V<sub>b</sub>∈R<sup><I>W×H×C</I></sup>。<I>W，H，C</I>分别是视频帧的像素宽，高和颜色通道数。参考另外俩paper【65，35】的协同注意力机制来发掘这两帧特征的相关性。具体说，计算V<sub>a</sub>和V<sub>b</sub>的**相似性矩阵S**</br>

![image](https://user-images.githubusercontent.com/11287531/115958848-5914a980-a55d-11eb-915a-fb47b7e2c066.png)

这里W∈R<sup><I>C×C</I></sup>是个权重矩阵。V<sub>a</sub>和V<sub>b</sub>的每一列表示的是一个C维的feature的向量，这样的feature一共有<I>WxH</I>个。这里V<sub>a</sub>，V<sub>b</sub>被flatten（摊平，降维）后，V<sub>a</sub>，V<sub>b</sub>∈R<sup><I>C×(WH)</I></sup>。S的每一个元素，表示的是V<sub>a</sub>和V<sub>b</sub>的每个feature的相似度。</br>

```diff
+ W的目的和意义是什么？W只是C×C，表示的是每个Channel的权重？比较两个图片的相似性，Channel权重重要吗？
```

因为W是一个方阵，它可以对角化为

![image](https://user-images.githubusercontent.com/11287531/115958861-65990200-a55d-11eb-94d4-a5f626cb35e7.png)

```diff
+ 1.为什么权重矩阵W可以对角化？
+ 2.我怀疑这个公式正确性。如果D是矩阵W的特征值对角矩阵，P是W的特征向量矩阵，方阵W对角化的话是不是应该写成下边斜体这样？我哪里错了吗？
 ```
*P<sup>-1</sup> W P = D</br>
=> (P P<sup>-1</sup>) W P = P D</br>
=>  W P = P D</br>
=>  W = P D P<sup>-1</sup></br>
而不是 W = P<sup>-1</sup> D P (公式2)。*

![image](https://user-images.githubusercontent.com/11287531/115958874-721d5a80-a55d-11eb-908e-4c64f81f8a27.png)


**对称的 协同注意力**

如果我们进一步约束权重矩阵W为一个对称矩阵。因为对称矩阵的特征值不等的特征向量正交，所以特征向量组矩阵P就变成了一个正交矩阵。这个对称的协同注意力可以演变成
![image](https://user-images.githubusercontent.com/11287531/115957756-56fc1c00-a558-11eb-8ad1-01fe913367f2.png)

用我的推导而不是paper里的公式的话</br>
W = P D P<sup>-1</sup> </br>
S = V<sub>b</sub><sup>T</sup> P D P<sup>-1</sup> V<sub>a</sub> </br>
这就不是计算两个投影结果的相似度了。


**原话It indicates that we project the feature embeddings V<sub>a</sub> and V<sub>b</sub> into an orthogonal common space and maintain their norm of V<sub>a</sub> and V<sub>b</sub>.**
```diff
+ 它表明什么？没明白。
```

这个属性已经被证明 对排除不同通道间的相关性 非常有帮助【50】同时有助于提升网络泛化能力【3，48】


**基于通道的 协同注意力**

而且，方阵W的特征向量组矩阵P 可以被简化成一个单位矩阵，然后W变成一个对角矩阵。这样的话，W可以被进一步对角化成两个对角矩阵D<sub>a</sub>和D<sub>b</sub>
```diff
+ 原话Furthermore, the project matrix P can be simplified into an identity matrix I (i.e., without space transformation), and then the weight matrix W becomes a diagonal matrix. In this case, W (i.e., D) can be further diagonalized into two diagonal matrices Da and Db.
+ 这句话什么意思？没明白。
```
因此，公式3可以写成基于通道的 协同注意力
![image](https://user-images.githubusercontent.com/11287531/115958061-db02d380-a559-11eb-964c-94e32b818c60.png)

这个操作等同于在计算相似性前，先对V<sub>a</sub> and V<sub>b</sub>应用了基于通道的权重。
这对于减少通道相关的冗余很有帮助。和另外2个论文有点像【7，20】。


======Ablation study================================</br>
在ablation study过程中，我们提供了详细的实验细节来显示不同协同注意力机制的效果。
在得到相似度矩阵S之后，我们用softmax分别对S进行行方向的和列方向的归一化。
![image](https://user-images.githubusercontent.com/11287531/115958250-c70ba180-a55a-11eb-91f1-e9c1b9296004.png)

随后，特征V<sub>a</sub>关于特征V<sub>b</sub>的注意力的summary可以描述为
![image](https://user-images.githubusercontent.com/11287531/115958354-3b464500-a55b-11eb-8299-71a9d894dd37.png)
相似地，也可以计算特征V<sub>b</sub>关于特征V<sub>a</sub>的注意力的summary。

考虑到输入的两帧的潜在外观变化，遮挡还有背景噪音，我们应该给不同帧不同权重，而不是平等的对待所有协同注意力信息。
为此，引入了一种self-gate机制，可以为每个注意力摘要分配 协同注意力置信度。
这个gate的公式如下
![image](https://user-images.githubusercontent.com/11287531/115958598-36ce5c00-a55c-11eb-8838-fc670c65919c.png)
σ 是个逻辑sigmoid激活函数，w<sub>f</sub>是卷积权重，b<sub>f</sub>和bias。
f<sub>g</sub>确定了有多少来自于引用帧的信息被保留，可以被自动学习。
计算了gate置信度后，注意力的摘要可以描述成
![image](https://user-images.githubusercontent.com/11287531/115958732-b1977700-a55c-11eb-902d-e6714858ebc2.png)

然后，把这个协同注意力Z和原始的特征V连接起来
![image](https://user-images.githubusercontent.com/11287531/115958770-e3a8d900-a55c-11eb-9c94-7947dc884111.png)

最后，被协同注意力强化了的特征X被送往一个分割网络用来产生最终的mask。
