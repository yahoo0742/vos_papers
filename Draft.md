Recurrent network

(ECCV2018) Hongmei Song, Wenguan Wang, Sanyuan Zhao, Jianbing Shen, and Kin-Man Lam, Pyramid Dilated Deeper ConvLSTM for Video Salient Object Detection
https://github.com/shenjianbing/PDB-ConvLSTM
https://openaccess.thecvf.com/content_ECCV_2018/papers/Hongmei_Song_Pseudo_Pyramid_Deeper_ECCV_2018_paper.pdf
Pyramid dilated bidirectional ConvLSTM architecture, and CRF-based post-process Continued
![image](https://user-images.githubusercontent.com/11287531/116868153-5eab7700-ac62-11eb-90bb-e8a50ae19804.png)

Optical flow offers explicit motion information, but also incurs significant computational cost, which severely limits the applicability of current video saliency models. 

The model consists of two key components. The first one, named Pyramid Dilated Convolution (PDC) module, which consists of four dilated convolutions with different dilation rates for explicitly extracting spatial saliency features in the same size but with different receptive field sizes. This is similar to observing the image from different distances. The multi-sacle spatial features are then concatenated together with the original input feature to feed the second component, two parallel DB-ConvLSTMs that are both implemented with 3x3x32 kernels for interpreting the temporal features of video frames and fusing spatial and temporal features automatically. The output of the DB-ConvLSTM branches in PDB-ConvLSTM are further concatenated as a multi-scale spatiotemporal saliency feature to feed a 1x1x1 convolution layer with a sigmoid activation for generating a binary mask which is at last upsampled to the size of the video frame via bilinear interpolation. Compared to the previous shallow, parallel bi-directional feature extraction strategy [REF_2015 Convolutional LSTM network Shi, X., Chen, Z., Wang, H.,], the PDB-ConvLSTM improves it with a deeper and cascaded learning process to bidirectionally capture information of both the forward and backward frames in a video.


feature map F = ResNet(a 473x473x3 video frame), 60x60x2048</br>
T1 = DilatedConv1(F) </br>
T2 = DilatedConv2(F) </br>
T3 = DilatedConv3(F) </br>
T4 = DilatedConv4(F) </br>
Z = Concatenate(F, T1, T2, T3, T4) </br>
With the dilated convolution, it computes dense CNN features at various receptive field sizes.






(CVPR2018) Flow Guided Recurrent Neural Encoder for Video Salient Object Detection
https://openaccess.thecvf.com/content_cvpr_2018/papers/Li_Flow_Guided_Recurrent_CVPR_2018_paper.pdf
https://blog.csdn.net/weixin_38682454/article/details/88024351




(CVPR2019) Wang et al., Learning Unsupervised Video Object Segmentation through Visual Attention
Wenguan Wang, Hongmei Song
https://github.com/wenguanwang/AGS
https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Learning_Unsupervised_Video_Object_Segmentation_Through_Visual_Attention_CVPR_2019_paper.pdf
based on the CNN-convLSTM architecture, Visual attention-driven unsupervised VOS model.
![image](https://user-images.githubusercontent.com/11287531/116980492-675f8400-ad1a-11eb-81cf-5f9c9c070be3.png)
![image](https://user-images.githubusercontent.com/11287531/116980550-79412700-ad1a-11eb-9572-88c4e5124409.png)


-----------------
Two-stream network 

(ICCV2017) J. Cheng and Y.-H. Tsai and S. Wang and M.-H. Yang, SegFlow: Joint Learning for Video Object Segmentation and Optical Flow
https://github.com/JingchunCheng/SegFlow
https://sites.google.com/site/yihsuantsai/research/iccv17-segflow
https://arxiv.org/pdf/1709.06750.pdf

(ICCV2017) Pavel Tokmakov, Karteek Alahari, and Cordelia Schmid, LVO Learning video object segmentation with visual Memory
https://arxiv.org/pdf/1704.05737.pdf
https://ieeexplore.ieee.org/document/8237742
Integrate one stream with appearance information and a visual memory module based on C-GRU.
![image](https://user-images.githubusercontent.com/11287531/116866369-64538d80-ac5f-11eb-8095-06cdf2aa4fae.png)
![image](https://user-images.githubusercontent.com/11287531/116866644-e0e66c00-ac5f-11eb-958a-f37262b40604.png)


(AAAI2020) Zhou, Tianfei and Wang, Shunzhou and Zhou, Yi and Yao, Yazhou and Li, Jianwu and Shao, Ling, Motion-Attentive Transition for Zero-Shot Video Object Segmentation
https://github.com/tfzhou/MATNet
https://arxiv.org/pdf/2003.04253.pdf


(CVPR2017) Jain et al., FusionSeg :Learning to combine motion and appearance for fully automatic segmentation of generic objects in video
https://feedforward.github.io/blog/fusionseg/
https://www.cs.utexas.edu/~grauman/papers/fusionseg-cvpr2017.pdf
FSEG. Design a two-stream fully CNN to combine appearance and motion information.
![image](https://user-images.githubusercontent.com/11287531/116865685-391c6e80-ac5e-11eb-96a1-63f567b2de8c.png)



---------------
Bottom-up Top-down

(CVPR2017) Tokmakov et al., Learning Motion Patterns in Videos
http://thoth.inrialpes.fr/research/mpnet/
https://openaccess.thecvf.com/content_cvpr_2017/papers/Tokmakov_Learning_Motion_Patterns_CVPR_2017_paper.pdf
MP-Net. takes the optical flow field of two consecutive frames of a video sequence as input and produces per-pixel motion labels.
![image](https://user-images.githubusercontent.com/11287531/116866023-cc55a400-ac5e-11eb-887f-1f86a470560d.png)


--------------
Teacher-student adaption
(ICRA2019) Mennatullah Siam, Chen Jiang, Steven Lu, Laura Petrich, Mahmoud Gamal, Mohamed Elhoseiny, and Martin Jager sand, Video Segmentation using Teacher-Student Adaptation in a Human Robot Interaction (HRI) Setting
https://deepai.org/publication/video-segmentation-using-teacher-student-adaptation-in-a-human-robot-interaction-hri-setting
![image](https://user-images.githubusercontent.com/11287531/117377207-d90c1d80-af26-11eb-8c77-ad7222f9ce8d.png)



Co-attention

(CVPR2019) Xiankai Lu, Wenguan Wang, Chao Ma, Jianbing Shen, Ling Shao, Fatih Porikli, See More, Know More: Unsupervised Video Object Segmentation with Co-Attention Siamese Networks
https://github.com/carrierlxk/COSNet
https://openaccess.thecvf.com/content_CVPR_2019/papers/Lu_See_More_Know_More_Unsupervised_Video_Object_Segmentation_With_Co-Attention_CVPR_2019_paper.pdf
![image](https://user-images.githubusercontent.com/11287531/117377875-5f752f00-af28-11eb-9e7a-0234e1c4f829.png)
![image](https://user-images.githubusercontent.com/11287531/117378191-02c64400-af29-11eb-8cfe-e6f8b4f9d76e.png)
![image](https://user-images.githubusercontent.com/11287531/117378216-0eb20600-af29-11eb-85f3-77cc9033fb3c.png)
![image](https://user-images.githubusercontent.com/11287531/117378252-2093a900-af29-11eb-8aa1-d6fe506643c9.png)
![image](https://user-images.githubusercontent.com/11287531/117378279-2b4e3e00-af29-11eb-8ec5-fc8b6ab18286.png)

This method is implemented with a siamese network which consists of three cascaded parts and a CRF post-process. The three parts are a DeepLabv3 based feature embedding module, a co-attention module, and a segmentation module. The DeepLabv3 backbone is pretrained with MSRA10K and DUT saliency datasets.
In the training phase, the DeepLabv3 based siamese network takes two streams (a query frame and a random sampled reference frame from the same video) as input to build the feature representions. 
The output are then refined by the co-attention module which first learns the normalized similarities between the feature representations, second computes the attention summaries with the feature representations and the normoalized similarities, and lastly feeds the co-attnetion enhanced features to a 1x1 convolutional layer to weight information from different input frames. The vanilla co-attention module was implemented using a fully connected layer with 512x512 parameters, while the channel-wise co-attention is built on a Squeeze-and-Excitation-like module.
Following the co-attention module, the features are passed into a segmentation network that consists of four convolutional layers and a sigmoid activation to generate the binary mask for the query frame.
After all, the binary mask is post-processed by CRF.
In the alblation study from the paper, without the co-attention module, the mean region similarity drops 9.2% on DAVIS16, 5.5% on FBMS, and 7.6% on Youtube-Objects.


Graph neural network
(ICCV2019) Wang, Wenguan and Lu, Xiankai and Shen, Jianbing and Crandall, David J. and Shao, Ling, Zero-Shot Video Object Segmentation via Attentive Graph Neural Networks
https://github.com/carrierlxk/AGNN
https://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Zero-Shot_Video_Object_Segmentation_via_Attentive_Graph_Neural_Networks_ICCV_2019_paper.pdf
![image](https://user-images.githubusercontent.com/11287531/117380019-eaf0bf00-af2c-11eb-815e-ea9d3d96544d.png)
![image](https://user-images.githubusercontent.com/11287531/117380044-fb089e80-af2c-11eb-8abd-b624aa896ec9.png)
![image](https://user-images.githubusercontent.com/11287531/117380078-1a073080-af2d-11eb-82ee-2fa32430f332.png)
![image](https://user-images.githubusercontent.com/11287531/117380121-2ab7a680-af2d-11eb-93db-9e9d98db814e.png)

The first five convolution blocks of DeepLabV3 is used as
the backbone for feature extraction. For an input video I,
each frame Ii (with a resolution of 473×473) is represented as a node vi in the video graph G and associated with
an initial node state vi = h<sub>i</sub><sup>0</sup> ∈ R <sup>60×60×256</sup>. 
Then, after a total of K message passing iterations, for each node vi, apply the function of the equation below to obtain a corresponding segmentation prediction map Sˆ ∈ [0, 1]<sup>60×60</sup>. 
![image](https://user-images.githubusercontent.com/11287531/117380993-25f3f200-af2f-11eb-8c26-6642ee67e04b.png)









===================


(2018) Wang, W., Shen, J., Shao, L., Video Salient Object Detection via Fully Convolutional Networks
https://www.researchgate.net/publication/319950992_Video_Salient_Object_Detection_via_Fully_Convolutional_Networks




(ECCV2020) Mingmin Zhen, Shiwei Li, Lei Zhou, Jiaxiang Shang, Haoan Feng, Tian Fang, Long Quan, Learning Discriminative Feature with CRF for Unsupervised Video Object Segmentation
https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123720443.pdf
![image](https://user-images.githubusercontent.com/11287531/117386211-99e7c780-af3a-11eb-8b2e-1dd3ef80266e.png)
To recognize the target object, the author expects to achieve two essential goals: (i) the ability to extract foreground objects from the individual frame; (ii) the ability to keep consistency among the video frames by correlating the features of each input image with discriminative features, which is extracted from input images selected from the same video randomly. 
The proposed network takes several images as input. The shared feature encoder adopts the fully convolutional DeepLabv3 to extract features.
The feature maps are then fed into a 1×1 convolutional layer to reduce the feature map channel to 256.
The output feature maps are as input for the discriminative feature module(DFM) to extract the discriminative features.
The input feature maps and the D-features go through an attention module(ATM) to reconstruct new feature maps
In the end, through a 3x3 convolutional layer, a BN layer and a 1x1 convolutional layer by a sigmoid function, the final binary output is obtained.

For the DFM, all feature maps from the input images are concatenated to form a large feature map F with size Nxhxwxc and then reshaped as Nhw x c.
K-group scoring module to calculate scores of K groups for distinguishing the discriminative features from noisy features.
![image](https://user-images.githubusercontent.com/11287531/117386225-a409c600-af3a-11eb-8971-3111d3d30012.png)
For each group, the concated feature map F multiplies a weight matrix W∈R<sup>cx1</sup> to obtain the final score 
s<sub>i</sub><sup>k</sup> = Softmax(F<sub>i<sub> W<sub>k</sub>)
s<sub>i</sub> is the i<sup>th<sup> feature of F to represent the discriminability of the feature.
The final discriminative feature for k<sup>th</sup> group of total K groups is comnputed as
F'<sub>k</sub> = Sum(s<sub>i</sub> F<sub>i</sub>) ∈R<sup>1xc</sup> 
The weight matrix W<sub>k</sub> is updated at training step t by
![image](https://user-images.githubusercontent.com/11287531/117390768-e0412480-af42-11eb-8d37-56ac7d60259c.png)

The ATM is adopted to mine the correlations between the input images.
By following the idea from COSNet and AGNN, the ATM compute the attention matrix P 
P = reshape(F<sub>i</sub>, hwxc) W<sub>att</sub> F'<sub>T<sub> ∈R<sup>hwxK</sup> 
where W<sub>att</sub> ∈R<sup>cxc</sup> is a learnable weight matrix.
Each element of P indicates the similarity of the corresponding feature of F<sub>i</sub> and feature of F'.
![image](https://user-images.githubusercontent.com/11287531/117390978-301feb80-af43-11eb-8889-ae839ec24e23.png)
Then the new feature map is reconstructed as
F<sup>new</sup> = reshape(softmax(P)F', hxwxc)

![image](https://user-images.githubusercontent.com/11287531/117393240-cd7d1e80-af47-11eb-97b6-3c9abfbe36c1.png)

At last, The smoothness and consistency of the attention map are considered as a classification problem and solved
by CRF.
![image](https://user-images.githubusercontent.com/11287531/117393468-4e3c1a80-af48-11eb-8c7b-cf0b9d90fe53.png)

F<sup>new</sup> = F<sup>new</sup> * conv(F<sup>new</sup>)
F<sub>i</sub> = F<sub>i</sub> * conv(F<sub>i</sub>)

At last, concatenate F<sup>new</sup> and F<sub>i</sub> to feed to a 1x1 convolutional layer to get the binary mask.

![image](https://user-images.githubusercontent.com/11287531/117393736-de7a5f80-af48-11eb-919c-83b361711278.png)





==========
AAAI2021 F2Net: Learning to Focus on the Foreground for Unsupervised Video Object Segmentation
https://arxiv.org/pdf/2012.02534.pdf


we propose a novel Focus on Foreground Network (F2Net),
which delves into the intra-inter frame details for the foreground objects and thus effectively improve the segmentation
performance. 

Specifically, our proposed network consists of
three main parts: Siamese Encoder Module, Center Guiding
Appearance Diffusion Module, and Dynamic Information Fusion Module. 

Firstly, we take a siamese encoder to extract
the feature representations of paired frames (reference frame
and current frame). 

Then, a Center Guiding Appearance Diffusion Module is designed to capture the inter-frame feature
(dense correspondences between reference frame and current
frame), intra-frame feature (dense correspondences in current frame), and original semantic feature of current frame.

Specifically, we establish a Center Prediction Branch to predict the center location of the foreground object in current
frame and leverage the center point information as spatial
guidance prior to enhance the inter-frame and intra-frame
feature extraction, and thus the feature representation considerably focus on the foreground objects. 


Finally, we propose a Dynamic Information Fusion Module to automatically select relatively important features through three aforementioned different level features. 


considering center point can be taken as the spatial prior guidance (Zhou, Wang, and Krahenb ¨ uhl 2019; Zhou, Koltun, and Kr ¨ ahenb ¨ uhl 2020; ¨ Wang et al. 2020), we tend to firstly predict the center point of the primary object and then segment the mask from such point to its surroundings


Different from the common appearance matching based methods, we additionally establish a Center Prediction Branch to estimate the center location of the primary object. Then, we encode the predicted center point into a gauss map as the spatial guidance prior to enhance the intra-frame and inter-frame feature matching in our Center Guiding Appearance Diffusion Module, leading the model to focus on the foreground object. After the appearance matching process, we can get three kinds of information flows: inter-frame features, intra-frame features, and original semantic features of current frame. Instead of fusing these three features by simple concatenation like previous methods, an attention based Dynamic Information Fusion Module is developed to automatically select the most discriminative features across the three features, providing more optimal representations for final segmentation.


F2Net illustrated in Figure 2, which consists of three main parts: a Siamese Encoder Module, a Center Guiding Appearance Diffusion Module, and a Dynamic Information Fusion Module. Given a pair of frames, we first generate their embeddings by a siamese encoder. After that, the Center Guiding Appearance Diffusion Module first predicts the foreground object center point by a Center Prediction Branch, then encodes it into a gauss map as the spatial-prior condition during the appearance matching process. After appearance matching, we can get three kinds of features. At last, we apply a Dynamic Information Fusion Module to fuse different level matched features for final segmentation.

The siamese encoder takes a pair of RGB images as inputs, including a current frame It ∈ RW×H×3 and a reference frame I0 ∈ RW×H×3 . Here, we take the first frame I0 as the reference frame because ASSUMPTION it is guaranteed to contain the foreground objects. The backbone network of the siamese encoder is DeepLabv3. We denote the extracted embeddings of It, I0 as Vt,V0 ∈ R W/8 × H/8 ×C.


Center Prediction Branch. We transform the point prediction into a heatmap Ht estimation task, which consists of two main steps: Feature Upsampling and Heatmap Generation. In Feature Upsampling, we adopt an upsample module to efficiently merge features of different res-blocks (res2, res3, res4, res5) in different scales to enhance the information for high-resolution features.  In Heatmap Generation, we consider two aspects as shown in Figure 3: 1) Spatial clues from previous frame: we propagate previous gauss map Gt−1 to current frame for better locating the foreground (if the current frame is the first frame, we utilize a zero map as the previous gauss map).  2) Semantic information from current feature we can directly estimate the heatmap Ft on Ut by applying a 3 × 3 convolutional layer with ReLU, followed by a 1×1 convolutional layer. At last, we add this two estimation branch into one with a sigmoid function by:


To choose the best center point ot = (xt, yt) ∈ R 2 from Ht, we additionally consider the motion clues across the sequential frames for better accuracy. using n history object centers in previous frames. At last, we compute the distance from each candidate to pt, and choose the closest one as the final center ot of the foreground object in current frame.


Spatial-Prior Guided Appearance Matching. To determine the foreground object, there are two essential properties: 1) distinguishable in an individual frame (locally saliency), and 2) frequently appearing throughout the video sequence (globally consistent). To achieve the first goal, we apply a non-local operation (Wang et al. 2018) on the current feature Vt to locate the salient object in an intra-wise matching way. To achieve our second goal, we utilize another non-local operation on both current and the reference features Vt,V0 to capture the inter-frame correlated information for alleviating appearance drift. We also employ a skip-connection on current feature Vt to preserve the semantic information. All three features contain different level information. To focus on the foreground, we encode the predicted center point ot into a gauss map Gt as spatial guidance prior for the intra-frame and interframe appearance matching. Compared to other appearance matching methods, the key difference of our method is that the encoded feature representation is weighted by the gaussguided spatial prior. 


After that, we reconstruct feature Vˆ t in which the pixel embeddings are weighted according to their similarity with the foreground.

At last, we can get three kinds of information flows  1) initial current feature Vt, 2) intra-frame feature Vˆ t,intra for intra-frame discriminability and 3) inter-frame feature Vˆ t,inter for inter-frame consistency.


Dynamic Information Fusion Original current feature Vt only contains a coarse clue for inferring the target foreground object without a global view. Intra-frame feature Vˆ t,intra contains more accurate salient object information in current frame, but fails to address the appearance changes in a video sequence. Inter-frame feature Vˆ t,inter contains more contexts to adapt to the appearance changes of the foreground objects. Instead of directly fusing these three features by simple concatenation (Lu et al. 2019; Yang et al. 2019), we need to selectively aggregate them to generate more discriminative features.   Channel-wise attention. The basic idea of channel-wise attention is to use gates to control the information flows from three-level features in channel dimension. Spatial-wise attention. Such spatial guidance is injected into the attention mechanism to selectively focus on the foreground pixels. To strengthen the information on the spatial dimension.

After the channel-wise and spatial-wise attention modules, the information flows from three-level input features can be adaptively aggregated into one feature map. Like most of previous works, we employ a decoder which consists of two convolutional layers to generate the final binary mask result Rt ∈ RW×H.





Following (Wang et al. 2019; Lu et al. 2019), we adopt two alternated steps to train our model. In the static-image iteration, we utilize image saliency dataset MSRA10K (Cheng et al. 2014) to fine-tune the DeepLabV3 based feature embedding module and the Center Prediction Branch. 
Meanwhile, in the dynamic-video iteration, we train the whole model with the training set in DAVIS16 (Perazzi et al. 2016). During training the Center Guiding Appearance Diffusion module, we first separately train the Center Prediction Branch and Spatial-Prior Guided Appearance Matching in parallel in the first 20 epochs, where we feed the appearance matching module with the center point of ground truth. Then, in latter epochs, we feed the appearance matching module with the predicted point from the Center Prediction Branch to jointly train them.








(arXiv2017) Vijayanarasimahan et al., SfM-Net: Learning of Structure and Motion from Video
https://ui.adsabs.harvard.edu/abs/2017arXiv170407804V/abstract
https://arxiv.org/pdf/1704.07804v1.pdf
Geometry-aware CNN to predict depth, segmentation, camera and rigid object motions
![image](https://user-images.githubusercontent.com/11287531/116867773-c7deba80-ac61-11eb-8529-835c21cb0646.png)
![image](https://user-images.githubusercontent.com/11287531/116867814-d2994f80-ac61-11eb-9046-74c73f81148f.png)




![image](https://user-images.githubusercontent.com/11287531/116868176-6bc86600-ac62-11eb-9e1f-a839c076456a.png)
![image](https://user-images.githubusercontent.com/11287531/116868190-75ea6480-ac62-11eb-8af5-deee86e3d61a.png)

(CVPR2018) Li et al., Instance Embedding Transfer to Unsupervised Video Object Segmentation
https://openaccess.thecvf.com/content_cvpr_2018/papers/Li_Instance_Embedding_Transfer_CVPR_2018_paper.pdf
Transfer transferring the knowledge encapsulated in image-based instance embedding networks, and adapt the instance networks to video object segmentation. They propose a motion-based bilateral network, then a graph cut model is build to propagate the pixel-wise labels.
![image](https://user-images.githubusercontent.com/11287531/116868417-dda0af80-ac62-11eb-8d84-2d5922a5db44.png)
![image](https://user-images.githubusercontent.com/11287531/116868463-ee512580-ac62-11eb-9c0c-ca932b14e900.png)
![image](https://user-images.githubusercontent.com/11287531/116868483-f4df9d00-ac62-11eb-93f3-f8e6fd14a2e2.png)


(NIPS2018) Goel et al., Unsupervised Video Object Segmentation for Deep Reinforcement Learning
https://arxiv.org/pdf/1805.07780.pdf
https://www.youtube.com/watch?v=HSYf3SdvGf0
https://medium.com/@shreydiwanjain/overview-of-mi-and-unsupervised-video-object-segmentation-through-reinforcement-learning-talk-a1cc02db297b
deep reinforcement learning methods is proposed to automatically detect moving objects with the relevant information for action selection.
![image](https://user-images.githubusercontent.com/11287531/116868820-9b2ba280-ac63-11eb-80a9-85e7979bd826.png)
![image](https://user-images.githubusercontent.com/11287531/116868845-a54da100-ac63-11eb-81b6-2521b8e6c084.png)




(CVPR2020) Xiankai Lu1, Wenguan Wang2, Jianbing Shen, Yu-Wing Tai, David Crandall, Steven C. H. Hoi, Learning Video Object Segmentation from Unlabeled Videos
https://github.com/carrierlxk/MuG
https://arxiv.org/pdf/2003.05020.pdf
![image](https://user-images.githubusercontent.com/11287531/117253345-3bb7d780-ae9b-11eb-9d74-97921440509a.png)
![image](https://user-images.githubusercontent.com/11287531/117253403-4a05f380-ae9b-11eb-8446-3e9f4ea446c5.png)
For a training video V ∈ S containing T frames: V = { X<sub>t</sub> }<sub>t=1</sub><sup>T</sup>, its features are specified as {x<sub>t</sub>}<sub>t=1</sub><sup>T</sup>, obtained from a FCN feature extractor ϕ: x<sub>t</sub>=ϕ(X<sub>t</sub>)∈ R<sup>W×H×C</sup>. 4-granularity characteristics for guiding the learning of ϕ.

Frame Granularity: Fore-background Knowledge Understanding. Basic fore-background mask Q<sub>t</sub> ∈ {0, 1} <sup>W×H</sup> for each frame X<sub>t</sub> is initially from background prior based saliency model <i>Saliency detection via graph-based manifold ranking</i>(unsupervised learning setting), or CAM maps from  <i>Multi-source weak supervision for saliency detection</i> or <i>Learning deep features for discriminative localization</i>(weakly supervised learning setting). 
Loss function:![image](https://user-images.githubusercontent.com/11287531/117277225-c3a9db80-aeb3-11eb-8a7d-c24d3f532b91.png)
L<sub>CE</sub> is the cross-entropy loss, P<sub>t</sub> is the prediction mask, the output of ρ(x<sub>t</sub>) where ρ: R<sup>WxHxC</sup> → [0,1]<sup>WxH</sup> map frame feature to a mask. Input: single frame feature x<sub>t</sub>. ρ 1x1 convolutional layer with simoid activation

Short-Term Granularity:  Intra-Clip Coherence Modeling. 
![image](https://user-images.githubusercontent.com/11287531/117253428-51c59800-ae9b-11eb-8f18-02fec260669d.png)
Given two consecutive frames X<sub>t</sub> and X<sub>t+1</sub>, crop a patch p from X<sub>t</sub>, apply ϕ on p and X<sub>t+1</sub>, we get 2 feature maps ϕ(p)∈ R<sup>wxhxc</sup> and x<sub>t+1</sub>∈ R<sup>WxHxC</sup>. conduct a cross-correlation operation we get S, a sigmoid-normalized tracking response map.
![image](https://user-images.githubusercontent.com/11287531/117285063-ad078280-aebb-11eb-9fca-da53ba6a3aaf.png)
The peak value on S, the most similar spot between the 2 feature maps, is considered as the new location of p, p' in X<sub>t+1</sub>, then backward track p' to frame X<sub>t</sub>. 
![image](https://user-images.githubusercontent.com/11287531/117288440-9d8a3880-aebf-11eb-8a6f-237c34ad59b8.png)
G<sub>p</sub> p ∈ [0, 1]<sup>W×H</sup> is a Gaussian-shap map with the same center of p and variance to the size of p.

continue cropping and conducting cross-correlation between X<sub>t+1</sub> and X<sub>t+2</sub>, then back track to X<sub>t+1</sub> to the initial frame X<sub>t</sub>.
With above designs, ϕ captures the spatiotemporally local correspondence and is content-discriminative
(due to its cross-frame target re-identification nature)

Long-Term Granularity Analysis: Cross-Frame Semantic Matching. Capturing this property is essential for ϕ, as it makes ϕ robust to many challenges, such as appearance variants, shape deformations, object occlusions, etc. the authors cast crossframe correspondence learning as a dual-frame semantic
matching problem.
![image](https://user-images.githubusercontent.com/11287531/117253476-5b4f0000-ae9b-11eb-9a07-15d684573b64.png)
input: disordered frames X<sub>i</sub>, X<sub>j</sub> randomly sampled from video
compute similarity affinity A<sub>i,j</sub> between (ϕ (X<sub>i</sub>), ϕ (X<sub>j</sub>)) by a co-attention operation
![image](https://user-images.githubusercontent.com/11287531/117290365-d3c8b780-aec1-11eb-86fa-d37da5ce093a.png)
where x<sub>i</sub> ∈ R<sup>Cx(WH)</sup> and x<sub>j</sub> ∈ R<sup>Cx(WH)</sup> are flat matrix of the feature maps of the 2 frames.
column-wise softmax.
Another small NN : R<sup>(W×H)×(W×H)</sup> → R <sup>6</sup> to regress a geometric transformation 6-degree of freedom.(trans, rot, scale)
![image](https://user-images.githubusercontent.com/11287531/117291498-32dafc00-aec3-11eb-9297-626f10592102.png)


As seen in Fig.1(b), the fore-background knowledge from the saliency [70] or
CAM [73, 76] is ambiguous and noisy. Inspired by Bootstrapping [40], we apply an iterative training strategy: after training with the initial fore-background maps, we use our trained model to re-label the training data. With each
iteration, the learner bootstraps itself by mining better forebackground knowledge and then leading a better model.

![image](https://user-images.githubusercontent.com/11287531/117292266-1ab7ac80-aec4-11eb-8c65-79f5f23a9895.png)
β1 = 0.1, β2 = 0.02 and β3 = 0.5.
Once the model is trained, the learned representations ϕ can be used for ZVOS and O-VOS, with slight modifications.
![image](https://user-images.githubusercontent.com/11287531/117292587-81d56100-aec4-11eb-8aa0-b163838ed868.png)



================

Datasets and Metrics:
• DAVIS16 [45] is a challenging video object segmentation dataset which consists of 50 videos in total (30 for training and 20 for val) with pixel-wise annotations for every frame. Three evaluation criteria are used in this dataset, i.e., region similarity (Intersection-over-Union)J , boundary accuracy F, and time stability T </br> 
• Youtube-Objects [47] comprises 126 video sequences belonging to 10 object categories and contain more than 20,000 frames in total. Following its protocol, J is commonly for measuring the segmentation performance. </br>
• DAVIS17 [46] consists of 60 videos in the training set, 30 videos in the validation set and 30 videos in the test-dev set. Different from DAVIS2016 and Youtube-Objects, which only focus on object-level video object segmentation, DAVIS17 provides instance-level annotations. </br>
