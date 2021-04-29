本文旨在对最新的跟踪方法进行全面回顾，并将这些方法分类为不同的类别，并确定新的趋势。

首先，我们提供了现有的分类方法，包括无监督的VOS，半监督的VOS，交互式VOS，弱监督的VOS和基于分段的跟踪方法。
其次，我们提供了不同方法的技术特征的详细讨论和概述。
第三，我们总结了相关视频数据集的特征，并提供了多种评估指标。
最后，我们指出了一组作品，并得出了自己的结论

Unsupervised VOS [17, 48, 58, 75, 95, 101]
Interactive VOS [13, 23, 104, 114, 176]
Semi-supervised VOS [22, 77, 125, 135, 162]
Another supervised way, produce masks of objects given the [155, 206] or natural language expressions [84] of the video label
![image](https://user-images.githubusercontent.com/11287531/116533071-13792780-a935-11eb-8427-5a90f5a2c798.png)

VOS算法隐式处理跟踪过程。 也就是说，自下而上的方法使用时空运动和外观相似性以完全自动化的方式分割视频。
这些方法一次读取多个或所有图像帧，以充分利用多个帧的上下文，并分割精确的Object mask。

VOT ...

1.1 Challenges and issues

在[126，187]中给出了更详细的困难描述。

VOST已经取得了巨大的进步，这些主要基于它们如何处理视觉分割和跟踪中的以下问题而彼此不同：
（i）哪种应用方案适合VOST？
（ii）哪种对象表示（即点，超像素，面片和对象）适合VOS？
（iii）哪些图像功能适合VOST？
（iv）如何在VOST中模拟物体的运动？
（v）如何对基于CNN的VOS方法进行逐处理和后处理？
（vi）哪些数据集适合评估VOST，其特征是什么？

本次调查将VOS和VOT分为大类，并对一些代表性方法进行了全面回顾。此外，我们将讨论VOST新趋势，并希望为新方法提供一些有趣的想法。

1.2 Organization and contributions of this survey

Unsupervised VOS: According to discover primary objects using appearance and motion cues, in Sec. 2.1, we categorize them as 
background subtraction, 
point trajectory, 
over-segmentation, 
“object-like” segments, 
and convolutional neural networks based methods. 

![image](https://user-images.githubusercontent.com/11287531/116531960-d3fe0b80-a933-11eb-9c84-cb27de305fe4.png)
![image](https://user-images.githubusercontent.com/11287531/116532965-f5abc280-a934-11eb-9a00-d81359993e78.png)
Table 1，总结了一些对象表示，例如，pixel，superpixel，supervoxel和patch以及image features。

VOS的另外一些survey[47,126]

----------------
2 MAJOR METHODS

2.1 Unsupervised video object segmentation

通常，现有的方法假定要分割和跟踪的对象具有不同的运动或频繁出现在图像序列中。看看五组无监督方法。

2.1.1背景扣除。
早期的视频分割方法主要是基于几何的，并且仅限于特定的运动背景。
经典的背景减法模拟每个像素的背景外观，并将快速变化的像素视为前景。图像和背景模型中的任何重大变化都代表移动的对象。移动区域里的像素被标记起来以进行进一步处理。
一个连接算法用于估计与对象连接的区域。因此，以上过程称为背景减法。通过 构造一个背景模型，然后为每个输入帧找到与模型的偏差，可以实现VOS。

根据所用运动的大小，背景减法可以分为固定背景[44、61、151]，2D参数运动的背景[11、38、76、136]和 3D运动的背景[19、75,161]。


2.1.5 Convolutional neural networks (CNN)

近年来，有些方法用CNN做VOS。早期的主要VOS方法先使用complementary CNN生成显着对象[100]，然后传播视频对象和superpixel-based neighborhood reversible flow。

后来，有几种VOS方法以端到端的方式采用了深度CNN。在[43、159、160、166]中，这些方法构建了一个双分支CNN来VOS。

MP-Net [159]将视频序列的两个连续帧的光流作为输入，并生成每个像素的运动标签。

为了解决MP-Net框架对象外观特征的局限性，Tokmakov等[160]将 外观信息 和 一个 基于卷积Gated Recurrent Units（GRU）[193]的视觉存储模块 集成为一个stream。

FSEG [43]还提出了一种具有外观和光流运动的双stream网络。

SfM-Net [166]combines two streams motion and structure to learn object masks and motion models without mask annotations by differentiable rending *what*

Li等。 [101]基于 图像实例embedding网络的 知识，使 instance networks 适应VOS*what*。此外，他们提出了一个基于运动的双边网络，然后建立了一个graph cut模型来传播像素级标签。
 
[57]提出了一种深度强化学习方法，该方法可以 根据相关信息自动检测 运动对象。

最近，Song等[149]提出了一种使用金字塔扩张的双向ConvLSTM架构的视频salient object detection方法，并将其应用于UVOS。

然后，基于CNN-convLSTM体系结构，Wang等[181]提出了一种视觉注意力驱动的UVOS模型。另外，他们从DAVIS [126]，Youtube-Objects [131]和SegTrack v2 [98]数据集收集UVOS人类注意力数据。


这些方法利用saliency，语义，光流或运动的信息来生成主要对象，然后将其传播到帧的其余部分。但是，由于不同实例与动态背景之间的运动混淆，这些无人监督的方法无法分割特定的对象。
此外，这些无监督方法的问题在于，由于许多不相关的干扰object proposals，它们的计算量很大。

Tab 1中列出了一些主要的UVOS方法的定性比较。



![image](https://user-images.githubusercontent.com/11287531/116540709-a2d70880-a93e-11eb-9607-f3e7e92b421d.png)

