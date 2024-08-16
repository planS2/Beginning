# What we have done 
DeepFakeDetection简介  

## 论文清单

zty  
UniversalFakeDetect  
[[Paper]](https://arxiv.org/abs/2302.10174)  
[[Code]](https://github.com/WisconsinAIVision/UniversalFakeDetect)  
简要介绍：
代码复现环境要求：~~能否跑通~~

zyc   
1.Rethinking the Up-Sampling Operations in CNN-based Generative Network for Generalizable Deepfake Detection   
[[Paper]](https://arxiv.org/pdf/2312.10461)     
[[Code]](https://github.com/chuangchuangtan/NPR-DeepfakeDetection)   
简要介绍：提出邻近像素关系(NPR)，用以捕捉和表征源自上采样操作的通用结构伪影。从局部图像像素的角度探索上采样层的痕迹，提出了一种简单但有效的伪影表示，称为邻域像素关系（Neighboring Pixel Relationships，NPR），以实现广义的深度伪造检测。

2.Learning on Gradients: Generalized Artifacts Representation for GAN-Generated Images Detection  
[[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Tan_Learning_on_Gradients_Generalized_Artifacts_Representation_for_GAN-Generated_Images_Detection_CVPR_2023_paper.pdf)  
[[Code]](https://github.com/chuangchuangtan/LGrad?tab=readme-ov-file)  
简要介绍：实用与训练好的cnn框架，将图像转化为梯度，由数据依赖问题转化为模型依赖问题，最后在进行分类ai生成图像。因为预训练的cnn已经在大量数据集上训练过，因此提高了泛化能力。  

---


# 分享会总结
## 2024/7/25
1. 建议以后分享要说清楚复现论文的核心创新点以及网络框架
2.  期待能够听到改进和创新

### to do
- 根据教程配置好wsl2
- 稳扎一下基础 了解GAN，CLIP  （除了建议的两位佬，大家可以自行看看或者等着下次例会直接听）
- 咋没人分享综述呢，下周要是有人觉得自己看的那篇不大看得懂，不如就来个DeepFake Detection大总结，分享一下综述
- 关注一下DataWhale第二期的夏令营，里面有相关介绍（不过比较适合DL入门）
[DataWhale第二期DeepFake Detection 学习手册]
- 如果你觉得自己比较闲，或者不押数模，我还是建议参加一下全球算法精英赛，同样的A2，显然这个对于咱们更好拿奖，而且感受了一周，DeepFake Detection应该还算是有意思的吧
- 复现代码的要对应起来

## 2024/8/12                      
### 总结
1. CLIP的作用就是将text和image联系起来，先各自进行encoder操作，用各自嵌入空间的向量represent原本的text和image（我觉得这里是个关键点，这也能加深理解encoder编码器的作用），例会分享时，我发现分享人习惯性的喜欢说这个encoder操作叫map映射到另一个空间，感觉是从数学上来说是对的，其实我理解的encoder编码器作用就是学习一种映射关系，从而 使用encoder之后的tensor就能表示原本的信息。


2. 有两种情况下网络简单但是效果却很好：   
   一种是特征工程做的非常好，在数据预处理部分就做的很完善，这样即便后期使用一些简单的模型仍能达到很好的效果
   另一种就是，已经使用了在大型数据集上经过多次训练的预训练模型

### To Do                   
~~接下来一段时间，决定还是以一半分享些各个方向，一半用来了解些必读论文，总不能搞ai的，该领域必读的那些论文还没读过吧~~
1. 
[[Attention is all you need]](https://arxiv.org/abs/1706.03762)











PS:这个title算是个老梗了，现在成了 Money is all you need 























