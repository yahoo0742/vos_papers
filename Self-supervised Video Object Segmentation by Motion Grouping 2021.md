In this paper, we aim to exploit such cues for object segmentation in a self-supervised manner, i.e. zero human annotation is required for training. 

At a high level, we aim to exploit the common fate principle, with the basic assumption being that elements tend to be perceived as a group if they move in the same direction at the same rate (have similar optical flow). 

Specifically, we tackle the problem by training a generative model that decomposes the optical flow into foreground (object) and background layers, describing each as a homogeneous field, with discontinuities occurring only between layers. 

We adopt a variant of the Transformer [3], with the self-attention being replaced by slot attention [46], where iterative grouping and binding have been built into the architecture. With some critical architectural changes, we show that pixels undergoing similar motion are grouped together and assigned to the same layer.

---------------
**Contributions:**

first, we introduce a simple architecture for video object
segmentation by exploiting motions, using only optical flow
as input. 

Second, we propose a self-supervised proxy task that is used to train the architecture without any manual supervision. 

Third, we conduct thorough ablation studies on the components that are key to the success of our architecture, such as a consistency loss on optical flow computed from various frame gaps. 

Fourth, we evaluate the proposed architecture on public benchmarks (DAVIS2016 [55], SegTrackv2 [42], and FBMS59 [56]), outperforming previous state-of-the-art self-supervised models, with comparable performance to the supervised approaches. 

Moreover, we also evaluate on a camouflage dataset (MoCA [41]), demonstrating a significant performance improvement over the other self- and supervised approaches, highlighting the importance of motion cues, and the potential bias towards visual appearance in existing video segmentation models

--------------
**2 Related Work**
Video object segmentation

In recent literature [5, 10, 12, 16, 24, 25, 31, 34, 35, 39, 40, 51, 52, 53, 54, 57, 57, 71, 73, 74, 75, 77, 84, 84, 87], 
Despite being called unsupervised VOS, in practice, the popular methods to address such problems extensively rely on supervised training, for example, by using two-stream networks [16, 31, 54, 71] trained on large-scale external datasets. 

Optical flow computation

Deep learning methods allow efficient computation of optical flow, both in supervised learning on synthetic data [68, 69], and in the self-supervised [44, 45] setting. 

Flow has been useful for a wide range of problems, occasionally even used in lieu of appearance cues (RGB images) for tracking [62], pose estimation [19], representation learning [50], and motion segmentation [10].

Transformer
In this work, we take inspiration from a specific variant of self-attention, namely slot attention [46], which was demonstrated to be effective for learning objectcentric representations on synthetic datasets, e.g. CLEVR[32].

Object-centric representations 
In this paper, we are the first to demonstrate its use for object segmentation of realistic videos by exploiting motion, where the challenging nuances in visual appearance (e.g. the complex background textures) have been removed.

---------------
**3 Method**
Input: an optical flow frame
Output: 1. a background layer representation + background weighted mask 2. a foreground layer containing one or more moving objects + weighted masks(oppacity layers)


