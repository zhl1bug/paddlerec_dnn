# [AI训练营]基于PaddleRec-dnn模型实现CTR点击率预测

基于dnn模型，进行点击率预测，商品数量有限，在点击率数据下，展示更多的推荐排序。

**[做个食谱点击排行版 后期会出小程序]**

# 一、项目背景

总所周知，推荐系统需要做，召回-粗排  排序-精排 等一些列的过程。

本次例子，选用排序-精排，DNN模型。

```
最开始的初衷是找个比较好理解的场景来做项目，快速学习深度学习。衣食住行中的食中入手。

1.需要做一个食谱的小程序，稍后会开另外一个项目，完成后会贴链接回来。

2.缺少用户评价数据，或者点击数据。三军未动，粮草先行，大数据作为粮草，是不可或缺的训练样本。那训练出来数据的好坏，需要有用户来进行评价。

3.所以才会有了当前的项目，基于用户的点击率，来对商品进行排序。
```

![](https://ai-studio-static-online.cdn.bcebos.com/ffd64add0b8d44869d1743b805fc5929a503f45e0ba24fdc9f076c5b56d857c8)
![](https://ai-studio-static-online.cdn.bcebos.com/2b5a0f61d67149e9bc0ce5c92856db550a00d1ee118445f1bd2b256a0b93fc11)
![](https://ai-studio-static-online.cdn.bcebos.com/a1c8b0d2fdd048ffaee6f7a30e62304d6227ab52b63c4d7282fe6a92d03de946)
![](https://ai-studio-static-online.cdn.bcebos.com/a5c44b76f92c45c8b5aafd23a41d43c36c8ed5f80f744da2b17d1927b958eb36)



# PaddleRec推荐系统相关资料

[十分钟！全流程！从零搭建推荐系统](https://aistudio.baidu.com/aistudio/projectdetail/559336?channelType=0&channel=0)

[告别电影荒，手把手教你训练符合自己口味的私人电影推荐助手](https://aistudio.baidu.com/aistudio/projectdetail/1481839?channelType=0&channel=0)

[PaddleRec gitee地址](https://gitee.com/paddlepaddle/PaddleRec)

[PaddleRec github地址](https://github.com/PaddlePaddle/PaddleRec)


# 进度表导图
```
当前除于重要的排序阶段，接下来，就开始深入了解DNN排序。
```
![](https://ai-studio-static-online.cdn.bcebos.com/93342d2592ad400c9edfd13e21307064dd2681a417ec43c3bb5e955212623149)




```python
# 1. 克隆PaddleRec项目

git clone https://gitee.com/PaddlePaddle/PaddleRec/
cd PaddleRec
```

[本次选用DNN模型，点击查看官方文档](https://gitee.com/paddlepaddle/PaddleRec/tree/master/models/rank/dnn)

接下来会用官方示例给的几个项目，给大家讲解如何使用。

## 数据集准备
官方自备了数据集，本次示例采用官方提供的Criteo作为测试


```python
# 查看官方数据集
ls PaddleRec/datasets/
```

    ag_news     BQ_simnet  FourSquare   LFM_1b_UGP		     readme.md
    ali-ccp     census     __init__.py  MIND		     Retailrocket
    AmazonBook  criteo     Jester	    movielens_pinterest_NCF  senti_clas
    Anime	    criteo_lr  letor07	    Netflix		     Steam
    BQ_dssm     Douban     LFM_1b	    one_billion		     TaFeng



```python
# 查看Criteo数据集
ls PaddleRec/datasets/criteo/
```

    data_process.sh  download.sh  get_slot_data.py	run.sh


## 快速开始

官方提供了快速开始的demo，让我们来尝试一下。


```python
# 进入模型目录
# cd models/rank/dnn # 在任意目录均可运行
# 动态图训练
python -u PaddleRec/tools/trainer.py -m PaddleRec/models/rank/dnn/config.yaml
```


```python
# 动态图预测
python -u PaddleRec/tools/infer.py -m PaddleRec/models/rank/dnn/config.yaml
```


```python
# 静态图训练
python -u PaddleRec/tools/static_trainer.py -m PaddleRec/models/rank/dnn/config.yaml  # 全量数据运行config_bigdata.yaml 
```


```python
# 静态图预测

python -u PaddleRec/tools/static_infer.py -m PaddleRec/models/rank/dnn/config.yaml
```

# 模型组网部分，引用官方示例

## 数据输入声明
正如数据准备章节所介绍，Criteo数据集中，分为连续数据与离散（稀疏）数据，所以整体而言，CTR-DNN模型的数据输入层包括三个，分别是：dense_input用于输入连续数据，维度由超参数dense_input_dim指定，数据类型是归一化后的浮点型数据。sparse_inputs用于记录离散数据，在Criteo数据集中，共有26个slot，所以我们创建了名为1~26的26个稀疏参数输入，数据类型为整数；最后是每条样本的label，代表了是否被点击，数据类型是整数，0代表负样例，1代表正样例。

## CTR-DNN模型组网
CTR-DNN模型的组网比较直观，本质是一个二分类任务，代码参考net.py。模型主要组成是一个Embedding层，四个FC层，以及相应的分类任务的loss计算和auc计算。

## Embedding层
首先介绍Embedding层的搭建方式：Embedding层的输入是sparse_input，由超参的sparse_feature_number和sparse_feature_dimshape定义。需要特别解释的是is_sparse参数，当我们指定is_sprase=True后，计算图会将该参数视为稀疏参数，反向更新以及分布式通信时，都以稀疏的方式进行，会极大的提升运行效率，同时保证效果一致。

各个稀疏的输入通过Embedding层后，将其合并起来，置于一个list内，以方便进行concat的操作。

self.embedding = paddle.nn.Embedding(
            self.sparse_feature_number,
            self.sparse_feature_dim,
            sparse=True,
            weight_attr=paddle.ParamAttr(
                name="SparseFeatFactors",
                initializer=paddle.nn.initializer.Uniform()))
## FC层
将离散数据通过embedding查表得到的值，与连续数据的输入进行concat操作，合为一个整体输入，作为全链接层的原始输入。我们共设计了4层FC，每层FC的输出维度由超参fc_sizes指定，每层FC都后接一个relu激活函数，每层FC的初始化方式为符合正态分布的随机初始化，标准差与上一层的输出维度的平方根成反比。

sizes = [sparse_feature_dim * num_field + dense_feature_dim
            ] + self.layer_sizes + [2]
acts = ["relu" for _ in range(len(self.layer_sizes))] + [None]
self._mlp_layers = []
for i in range(len(layer_sizes) + 1):
    linear = paddle.nn.Linear(
        in_features=sizes[i],
        out_features=sizes[i + 1],
        weight_attr=paddle.ParamAttr(
            initializer=paddle.nn.initializer.Normal(
                std=1.0 / math.sqrt(sizes[i]))))
    self.add_sublayer('linear_%d' % i, linear)
    self._mlp_layers.append(linear)
    if acts[i] == 'relu':
        act = paddle.nn.ReLU()
        self.add_sublayer('act_%d' % i, act)
        self._mlp_layers.append(act)
## Loss及Auc计算
预测的结果通过一个输出shape为2的FC层给出，该FC层的激活函数是softmax，会给出每条样本分属于正负样本的概率。
每条样本的损失由交叉熵给出。
我们同时还会计算预测的auc。

# 效果复现

为了方便使用者能够快速的跑通每一个模型，我们在每个模型下都提供了样例数据。如果需要复现readme中的效果,请按如下步骤依次操作即可。 在全量数据下模型的指标如下：


| 模型 | auc | batch_size | epoch_num | Time of each epoch |
| -------- | -------- | -------- | -------- |-------- |
| dnn     | 0.7748     | 512     | 4 | 约3小时 |


确认您当前所在目录为PaddleRec/models/rank/dnn
进入paddlerec/datasets/criteo目录下，执行该脚本，会从国内源的服务器上下载我们预处理完成的criteo全量数据集，并解压到指定文件夹。


```python
# 下载criteo数据集  全量数据集，下载需要时间
sh PaddleRec/datasets/criteo/run.sh
```


```python
#查看config配置文件

cat PaddleRec/models/rank/dnn/config_bigdata.yaml
```

# 修改config

```
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# workspace
#workspace: "models/rank/dnn"


runner:
  train_data_dir: "../../../datasets/criteo/slot_train_data_full"
  train_reader_path: "criteo_reader" # importlib format
  use_gpu: False
  use_auc: True
  train_batch_size: 512
  epochs: 4
  print_interval: 10
  model_save_path: "output_model_dnn_all"
  infer_reader_path: "criteo_reader" # importlib format
  test_data_dir: "../../../datasets/criteo/slot_test_data_full"
  infer_batch_size: 512
  infer_load_path: "output_model_dnn_all"
  infer_start_epoch: 0
  infer_end_epoch: 4

  #thread_num: 5
  #reader_type: "QueueDataset"  # DataLoader / QueueDataset / RecDataset
  #pipe_command: "python3.7 queuedataset_reader.py"
  #dataset_debug: False
  #split_file_list: False

# hyper parameters of user-defined network
hyper_parameters:
  # optimizer config
  optimizer:
    class: Adam
    learning_rate: 0.001
    strategy: async
  # user-defined <key, value> pairs
  sparse_inputs_slots: 27
  sparse_feature_number: 1000001
  sparse_feature_dim: 9
  dense_input_dim: 13
  fc_sizes: [512, 256, 128, 32]
  distributed_embedding: 0

```

# config参数详解

选用几个实用的，其他具体的看config文件，你已经是个成熟的开发了， 应该要学会查看配置文件，并且修改参数。

// train_data_dir 训练数据集位置

// use_gpu  是否实用gpu

// use_auc  auc开启

// train_batch_size 训练大小限制

// model_save_path 模型保存名称

// test_data_dir 预测训练集位置

// infer_batch_size 预测大小限制

// infer_load_path 预测模型加载路径




# 此处使用 Embedding，所以对应的知识，往回翻阅一下 Embedding层的知识

// optimizer 优化器参数

// learning_rate 学习率

// sparse_inputs_slots  sparse的数量

// sparse_feature_number  sparse_feature的数量

// sparse_feature_dim  shape的超参

// dense_input_dim  输入数量

// fc_sizes fc输出维度

// distributed_embedding 分布式嵌入



```python
# 将前面下载的 slot_train_data_full、slot_test_data_full 移动到训练集下

mv slot_test_data_full/ PaddleRec/datasets/criteo/
mv slot_train_data_full/ PaddleRec/datasets/criteo/
```


```python
# 动态图训练 需要相对较长的时间
python -u PaddleRec/tools/trainer.py -m PaddleRec/models/rank/dnn/config_bigdata.yaml  # 全量数据运行config_bigdata.yaml 
```


```python
# 动态图预测
python -u PaddleRec/tools/infer.py -m PaddleRec/models/rank/dnn/config_bigdata.yaml  # 全量数据运行config_bigdata.yaml 
```

# 得到训练以后的模型

根据默认config参数
output_model_dnn/ 文件夹下的文件，就是训练之后的模型

# 使用模型部署
[PaddleHub教程合集](https://aistudio.baidu.com/aistudio/projectdetail/231146?channelType=0&channel=0)

[PaddleHub gitee](https://gitee.com/PaddlePaddle/PaddleHub)

[PaddleHub github](https://github.com/PaddlePaddle/PaddleHub)

# 五、总结与升华

PaddleRec开箱即用，给推荐系统的排序，带来了便利。后续会做多个推荐系统相关的项目，加入实用场景。

# 个人简介

张宏理，厦门飞桨领航团团长，在厦门的小伙伴，请联系我，一起加入我们。

[AI Studio链接： https://aistudio.baidu.com/aistudio/personalcenter/thirdview/816197](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/816197)


[个人博客:holyzhang.com](http://www.holyzhang.com/)

[github:https://github.com/zzzhanghongli](https://github.com/zzzhanghongli)

[gitee:https://gitee.com/holyz](https://gitee.com/holyz)
