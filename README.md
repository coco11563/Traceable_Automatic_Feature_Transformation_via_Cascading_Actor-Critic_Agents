# Traceable Automatic Feature Transformation via Cascading Actor-Critic Agents
## Basic info:
This is the release code for :
[Traceable Automatic Feature Transformation via Cascading Actor-Critic Agents](https://arxiv.org/abs/2212.13402) 
which is accepted by SDM 2023!

Recommended ref:
```
Meng Xiao, Dongjie Wang, Min Wu, Ziyue Qiao, Pengfei Wang, Kunpeng Liu, Yuanchun Zhou, Yanjie Fu. Traceable Automatic Feature Transformation via Cascading Actor-Critic Agents. SIAM International Conference on Data Mining 2023, 2023
```

Recommended Bib:
```
@article{wang2022group,
  title={Traceable Automatic Feature Transformation via Cascading Actor-Critic Agents},
  author = {Xiao, Meng and Wang, Dongjie and Wu, Min and Qiao, Ziyue and Wang, Pengfei and Liu, Kunpeng and Zhou, Yuanchun and Fu, Yanjie},
  journal={SIAM International Conference on Data Mining 2023},
  year={2023}
}
```
***
## Paper Abstract
Feature transformation for AI is an essential task to boost the effectiveness and interpretability of machine learning (ML). Feature transformation aims to transform original data to identify an optimal feature space that enhances the performances of a downstream ML model. Existing studies either combines preprocessing, feature selection, and generation skills to empirically transform data,  or automate feature transformation by machine intelligence, such as reinforcement learning. However, existing studies suffer from: 1) high-dimensional non-discriminative feature space; 2) inability to represent complex situational states;  3) inefficiency in integrating local and global feature information. To fill the research gap, we propose a novel group-wise cascading actor-critic perspective to develop the AI construct of automated feature transformation. 
Specifically, we formulate the feature transformation task as an iterative, nested process of feature generation and selection, where feature generation is to generate and add new features based on original features, and feature selection is to remove redundant features to control the size of feature space. Our proposed framework has three technical aims: 1) efficient generation; 2) effective policy learning; 3) accurate state perception. For an efficient generation, we develop a tailored feature clustering algorithm and accelerate generation by feature group-group crossing based generation. For effective policy learning, we propose a cascading actor-critic learning strategy to learn state-passing agents to select candidate feature groups and operations for fast feature generation. Such a strategy can effectively learn policies when the original feature size is large, along with exponentially growing feature generation action space, in which classic Q-value estimation methods fail. For accurate state perception of feature space, we develop a state comprehension method considering not only pointwise feature information but also pairwise feature-feature correlations. Finally, we present extensive experiments and case studies to illustrate 24.7\% improvements in F1 scores compared with SOTAs and robustness in high-dimensional data.
***


![image](https://user-images.githubusercontent.com/13342088/209976599-863e0586-0748-4e7a-9458-c21d34fe0831.png)


## How to run:
### step 1: download the code and dataset:
```
git clone git@github.com:coco11563/Traceable_Automatic_Feature_Transformation_via_Cascading_Actor-Critic_Agents.git
```
then:
```
follow the instruction in readme.md in `/data/processed/data_info.md` to get the dataset
```

### step 2: run the code with main script:`main.py`

```
xxx/python3 main.py --name DATASETNAME --episodes SEARCH_EP_NUM --steps SEARCH_STEP_NUM...
```

please check each configuration in `initial.py`

### step 3: enjoy the generated dataset:

the generated feature will in ./tmp/NAME_SAMPLINE_METHOD_/xxx.csv
