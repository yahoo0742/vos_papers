(CVPR2017) Jain et al., FusionSeg :Learning to combine motion and appearance for fully automatic segmentation of generic objects in video
https://feedforward.github.io/blog/fusionseg/
https://www.cs.utexas.edu/~grauman/papers/fusionseg-cvpr2017.pdf
FSEG. Design a two-stream fully CNN to combine appearance and motion information.
![image](https://user-images.githubusercontent.com/11287531/116865685-391c6e80-ac5e-11eb-96a1-63f567b2de8c.png)

(CVPR2017) Tokmakov et al., Learning Motion Patterns in Videos
http://thoth.inrialpes.fr/research/mpnet/
https://openaccess.thecvf.com/content_cvpr_2017/papers/Tokmakov_Learning_Motion_Patterns_CVPR_2017_paper.pdf
MP-Net. takes the optical flow field of two consecutive frames of a video sequence as input and produces per-pixel motion labels.
![image](https://user-images.githubusercontent.com/11287531/116866023-cc55a400-ac5e-11eb-887f-1f86a470560d.png)

(arXiv2017) Tokmakov et al., LVO Learning video object segmentation with visual Memory
https://arxiv.org/pdf/1704.05737.pdf
https://ieeexplore.ieee.org/document/8237742
Integrate one stream with appearance information and a visual memory module based on C-GRU.
![image](https://user-images.githubusercontent.com/11287531/116866369-64538d80-ac5f-11eb-8095-06cdf2aa4fae.png)
![image](https://user-images.githubusercontent.com/11287531/116866644-e0e66c00-ac5f-11eb-958a-f37262b40604.png)

(arXiv2017) Vijayanarasimahan et al., SfM-Net: Learning of Structure and Motion from Video
https://ui.adsabs.harvard.edu/abs/2017arXiv170407804V/abstract
https://arxiv.org/pdf/1704.07804v1.pdf
Geometry-aware CNN to predict depth, segmentation, camera and rigid object motions
![image](https://user-images.githubusercontent.com/11287531/116867773-c7deba80-ac61-11eb-8529-835c21cb0646.png)
![image](https://user-images.githubusercontent.com/11287531/116867814-d2994f80-ac61-11eb-9046-74c73f81148f.png)

(ECCV2018) Song et al., Pyramid Dilated Deeper ConvLSTM for Video Salient Object Detection
https://github.com/shenjianbing/PDB-ConvLSTM
https://openaccess.thecvf.com/content_ECCV_2018/papers/Hongmei_Song_Pseudo_Pyramid_Deeper_ECCV_2018_paper.pdf
Pyramid dilated bidirectional ConvLSTM architecture, and CRF-based post-process Continued
![image](https://user-images.githubusercontent.com/11287531/116868153-5eab7700-ac62-11eb-90bb-e8a50ae19804.png)
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


(CVPR2019) Wang et al., Learning Unsupervised Video Object Segmentation through Visual Attention
Wenguan Wang, Hongmei Song
https://github.com/wenguanwang/AGS
https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Learning_Unsupervised_Video_Object_Segmentation_Through_Visual_Attention_CVPR_2019_paper.pdf
based on the CNN-convLSTM architecture, Visual attention-driven unsupervised VOS model.
![image](https://user-images.githubusercontent.com/11287531/116980492-675f8400-ad1a-11eb-81cf-5f9c9c070be3.png)
![image](https://user-images.githubusercontent.com/11287531/116980550-79412700-ad1a-11eb-9572-88c4e5124409.png)


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


Video Object Segmentation and Tracking: A Survey
https://eungbean.github.io/2019/09/04/awesome-video-object-segmentation/
