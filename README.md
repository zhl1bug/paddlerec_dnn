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

    Cloning into 'PaddleRec'...
    remote: Enumerating objects: 18304, done.[K
    remote: Total 18304 (delta 0), reused 0 (delta 0), pack-reused 18304[K
    Receiving objects: 100% (18304/18304), 76.82 MiB | 5.09 MiB/s, done.
    Resolving deltas: 100% (10595/10595), done.
    Checking connectivity... done.


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

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/setuptools/depends.py:2: DeprecationWarning: the imp module is deprecated in favour of importlib; see the module's documentation for alternative uses
      import imp
    2021-08-15 12:26:29,370 - INFO - **************common.configs**********
    2021-08-15 12:26:29,370 - INFO - use_gpu: False, use_visual: False, train_batch_size: 2, train_data_dir: data/sample_data/train, epochs: 3, print_interval: 2, model_save_path: output_model_dnn
    2021-08-15 12:26:29,370 - INFO - **************common.configs**********
    2021-08-15 12:26:29,499 - INFO - read data
    2021-08-15 12:26:29,500 - INFO - reader path:criteo_reader
    2021-08-15 12:26:29,716 - INFO - epoch: 0, batch_id: 0, auc:0.000000,  avg_reader_cost: 0.00189 sec, avg_batch_cost: 0.10502 sec, avg_samples: 1.00000, ips: 9.52174 ins/s
    2021-08-15 12:26:29,952 - INFO - epoch: 0, batch_id: 2, auc:0.400000,  avg_reader_cost: 0.00030 sec, avg_batch_cost: 0.11586 sec, avg_samples: 2.00000, ips: 17.26248 ins/s
    2021-08-15 12:26:30,187 - INFO - epoch: 0, batch_id: 4, auc:0.312500,  avg_reader_cost: 0.00030 sec, avg_batch_cost: 0.11516 sec, avg_samples: 2.00000, ips: 17.36729 ins/s
    2021-08-15 12:26:30,420 - INFO - epoch: 0, batch_id: 6, auc:0.363636,  avg_reader_cost: 0.00031 sec, avg_batch_cost: 0.11468 sec, avg_samples: 2.00000, ips: 17.43961 ins/s
    2021-08-15 12:26:30,655 - INFO - epoch: 0, batch_id: 8, auc:0.533333,  avg_reader_cost: 0.00030 sec, avg_batch_cost: 0.11494 sec, avg_samples: 2.00000, ips: 17.40104 ins/s
    2021-08-15 12:26:30,889 - INFO - epoch: 0, batch_id: 10, auc:0.458333,  avg_reader_cost: 0.00028 sec, avg_batch_cost: 0.11474 sec, avg_samples: 2.00000, ips: 17.43107 ins/s
    2021-08-15 12:26:31,123 - INFO - epoch: 0, batch_id: 12, auc:0.447619,  avg_reader_cost: 0.00030 sec, avg_batch_cost: 0.11475 sec, avg_samples: 2.00000, ips: 17.42919 ins/s
    2021-08-15 12:26:31,358 - INFO - epoch: 0, batch_id: 14, auc:0.512000,  avg_reader_cost: 0.00031 sec, avg_batch_cost: 0.11516 sec, avg_samples: 2.00000, ips: 17.36650 ins/s
    2021-08-15 12:26:31,593 - INFO - epoch: 0, batch_id: 16, auc:0.470238,  avg_reader_cost: 0.00030 sec, avg_batch_cost: 0.11517 sec, avg_samples: 2.00000, ips: 17.36520 ins/s
    2021-08-15 12:26:31,874 - INFO - epoch: 0, batch_id: 18, auc:0.515625,  avg_reader_cost: 0.00035 sec, avg_batch_cost: 0.13694 sec, avg_samples: 2.00000, ips: 14.60452 ins/s
    2021-08-15 12:26:32,111 - INFO - epoch: 0, batch_id: 20, auc:0.465306,  avg_reader_cost: 0.00050 sec, avg_batch_cost: 0.11601 sec, avg_samples: 2.00000, ips: 17.24047 ins/s
    2021-08-15 12:26:32,346 - INFO - epoch: 0, batch_id: 22, auc:0.384384,  avg_reader_cost: 0.00034 sec, avg_batch_cost: 0.11528 sec, avg_samples: 2.00000, ips: 17.34911 ins/s
    2021-08-15 12:26:32,581 - INFO - epoch: 0, batch_id: 24, auc:0.400000,  avg_reader_cost: 0.00033 sec, avg_batch_cost: 0.11507 sec, avg_samples: 2.00000, ips: 17.38143 ins/s
    2021-08-15 12:26:32,816 - INFO - epoch: 0, batch_id: 26, auc:0.439746,  avg_reader_cost: 0.00033 sec, avg_batch_cost: 0.11529 sec, avg_samples: 2.00000, ips: 17.34739 ins/s
    2021-08-15 12:26:33,052 - INFO - epoch: 0, batch_id: 28, auc:0.530233,  avg_reader_cost: 0.00035 sec, avg_batch_cost: 0.11594 sec, avg_samples: 2.00000, ips: 17.25032 ins/s
    2021-08-15 12:26:33,288 - INFO - epoch: 0, batch_id: 30, auc:0.529891,  avg_reader_cost: 0.00034 sec, avg_batch_cost: 0.11565 sec, avg_samples: 2.00000, ips: 17.29389 ins/s
    2021-08-15 12:26:33,523 - INFO - epoch: 0, batch_id: 32, auc:0.523409,  avg_reader_cost: 0.00034 sec, avg_batch_cost: 0.11523 sec, avg_samples: 2.00000, ips: 17.35661 ins/s
    2021-08-15 12:26:33,760 - INFO - epoch: 0, batch_id: 34, auc:0.586006,  avg_reader_cost: 0.00034 sec, avg_batch_cost: 0.11586 sec, avg_samples: 2.00000, ips: 17.26216 ins/s
    2021-08-15 12:26:33,993 - INFO - epoch: 0, batch_id: 36, auc:0.573427,  avg_reader_cost: 0.00035 sec, avg_batch_cost: 0.11432 sec, avg_samples: 2.00000, ips: 17.49451 ins/s
    2021-08-15 12:26:34,227 - INFO - epoch: 0, batch_id: 38, auc:0.564427,  avg_reader_cost: 0.00034 sec, avg_batch_cost: 0.11475 sec, avg_samples: 2.00000, ips: 17.42944 ins/s
    2021-08-15 12:26:34,345 - INFO - epoch: 0 done, auc: 0.547674, epoch time: 4.84 s
    2021-08-15 12:26:34,724 - INFO - Already save model in output_model_dnn/0
    2021-08-15 12:26:34,849 - INFO - epoch: 1, batch_id: 0, auc:1.000000,  avg_reader_cost: 0.00140 sec, avg_batch_cost: 0.06038 sec, avg_samples: 1.00000, ips: 16.56093 ins/s
    2021-08-15 12:26:35,084 - INFO - epoch: 1, batch_id: 2, auc:1.000000,  avg_reader_cost: 0.00030 sec, avg_batch_cost: 0.11543 sec, avg_samples: 2.00000, ips: 17.32641 ins/s
    2021-08-15 12:26:35,319 - INFO - epoch: 1, batch_id: 4, auc:0.750000,  avg_reader_cost: 0.00032 sec, avg_batch_cost: 0.11532 sec, avg_samples: 2.00000, ips: 17.34364 ins/s
    2021-08-15 12:26:35,555 - INFO - epoch: 1, batch_id: 6, auc:0.878788,  avg_reader_cost: 0.00031 sec, avg_batch_cost: 0.11532 sec, avg_samples: 2.00000, ips: 17.34237 ins/s
    2021-08-15 12:26:35,790 - INFO - epoch: 1, batch_id: 8, auc:0.911111,  avg_reader_cost: 0.00034 sec, avg_batch_cost: 0.11538 sec, avg_samples: 2.00000, ips: 17.33330 ins/s
    2021-08-15 12:26:36,025 - INFO - epoch: 1, batch_id: 10, auc:0.833333,  avg_reader_cost: 0.00029 sec, avg_batch_cost: 0.11509 sec, avg_samples: 2.00000, ips: 17.37725 ins/s
    2021-08-15 12:26:36,260 - INFO - epoch: 1, batch_id: 12, auc:0.809524,  avg_reader_cost: 0.00030 sec, avg_batch_cost: 0.11523 sec, avg_samples: 2.00000, ips: 17.35703 ins/s
    2021-08-15 12:26:36,495 - INFO - epoch: 1, batch_id: 14, auc:0.840000,  avg_reader_cost: 0.00030 sec, avg_batch_cost: 0.11542 sec, avg_samples: 2.00000, ips: 17.32774 ins/s
    2021-08-15 12:26:36,743 - INFO - epoch: 1, batch_id: 16, auc:0.797619,  avg_reader_cost: 0.00031 sec, avg_batch_cost: 0.11578 sec, avg_samples: 2.00000, ips: 17.27371 ins/s
    2021-08-15 12:26:36,982 - INFO - epoch: 1, batch_id: 18, auc:0.822917,  avg_reader_cost: 0.00038 sec, avg_batch_cost: 0.11717 sec, avg_samples: 2.00000, ips: 17.06925 ins/s
    2021-08-15 12:26:37,217 - INFO - epoch: 1, batch_id: 20, auc:0.775510,  avg_reader_cost: 0.00034 sec, avg_batch_cost: 0.11533 sec, avg_samples: 2.00000, ips: 17.34187 ins/s
    2021-08-15 12:26:37,451 - INFO - epoch: 1, batch_id: 22, auc:0.684685,  avg_reader_cost: 0.00029 sec, avg_batch_cost: 0.11474 sec, avg_samples: 2.00000, ips: 17.43028 ins/s
    2021-08-15 12:26:37,686 - INFO - epoch: 1, batch_id: 24, auc:0.705000,  avg_reader_cost: 0.00032 sec, avg_batch_cost: 0.11515 sec, avg_samples: 2.00000, ips: 17.36855 ins/s
    2021-08-15 12:26:37,921 - INFO - epoch: 1, batch_id: 26, auc:0.733615,  avg_reader_cost: 0.00030 sec, avg_batch_cost: 0.11544 sec, avg_samples: 2.00000, ips: 17.32494 ins/s
    2021-08-15 12:26:38,156 - INFO - epoch: 1, batch_id: 28, auc:0.728682,  avg_reader_cost: 0.00028 sec, avg_batch_cost: 0.11485 sec, avg_samples: 2.00000, ips: 17.41329 ins/s
    2021-08-15 12:26:38,389 - INFO - epoch: 1, batch_id: 30, auc:0.751359,  avg_reader_cost: 0.00029 sec, avg_batch_cost: 0.11451 sec, avg_samples: 2.00000, ips: 17.46584 ins/s
    2021-08-15 12:26:38,623 - INFO - epoch: 1, batch_id: 32, auc:0.764706,  avg_reader_cost: 0.00027 sec, avg_batch_cost: 0.11465 sec, avg_samples: 2.00000, ips: 17.44447 ins/s
    2021-08-15 12:26:38,859 - INFO - epoch: 1, batch_id: 34, auc:0.779397,  avg_reader_cost: 0.00031 sec, avg_batch_cost: 0.11554 sec, avg_samples: 2.00000, ips: 17.30989 ins/s
    2021-08-15 12:26:39,093 - INFO - epoch: 1, batch_id: 36, auc:0.770105,  avg_reader_cost: 0.00030 sec, avg_batch_cost: 0.11476 sec, avg_samples: 2.00000, ips: 17.42771 ins/s
    2021-08-15 12:26:39,326 - INFO - epoch: 1, batch_id: 38, auc:0.754150,  avg_reader_cost: 0.00031 sec, avg_batch_cost: 0.11402 sec, avg_samples: 2.00000, ips: 17.54034 ins/s
    2021-08-15 12:26:39,444 - INFO - epoch: 1 done, auc: 0.732265, epoch time: 4.72 s
    2021-08-15 12:26:39,771 - INFO - Already save model in output_model_dnn/1
    2021-08-15 12:26:39,895 - INFO - epoch: 2, batch_id: 0, auc:1.000000,  avg_reader_cost: 0.00100 sec, avg_batch_cost: 0.05996 sec, avg_samples: 1.00000, ips: 16.67748 ins/s
    2021-08-15 12:26:40,130 - INFO - epoch: 2, batch_id: 2, auc:1.000000,  avg_reader_cost: 0.00030 sec, avg_batch_cost: 0.11498 sec, avg_samples: 2.00000, ips: 17.39502 ins/s
    2021-08-15 12:26:40,366 - INFO - epoch: 2, batch_id: 4, auc:0.937500,  avg_reader_cost: 0.00029 sec, avg_batch_cost: 0.11551 sec, avg_samples: 2.00000, ips: 17.31391 ins/s
    2021-08-15 12:26:40,600 - INFO - epoch: 2, batch_id: 6, auc:0.969697,  avg_reader_cost: 0.00029 sec, avg_batch_cost: 0.11477 sec, avg_samples: 2.00000, ips: 17.42597 ins/s
    2021-08-15 12:26:40,835 - INFO - epoch: 2, batch_id: 8, auc:0.977778,  avg_reader_cost: 0.00030 sec, avg_batch_cost: 0.11501 sec, avg_samples: 2.00000, ips: 17.38925 ins/s
    2021-08-15 12:26:41,069 - INFO - epoch: 2, batch_id: 10, auc:0.972222,  avg_reader_cost: 0.00031 sec, avg_batch_cost: 0.11482 sec, avg_samples: 2.00000, ips: 17.41918 ins/s
    2021-08-15 12:26:41,304 - INFO - epoch: 2, batch_id: 12, auc:0.971429,  avg_reader_cost: 0.00030 sec, avg_batch_cost: 0.11544 sec, avg_samples: 2.00000, ips: 17.32477 ins/s
    2021-08-15 12:26:41,540 - INFO - epoch: 2, batch_id: 14, auc:0.976000,  avg_reader_cost: 0.00032 sec, avg_batch_cost: 0.11541 sec, avg_samples: 2.00000, ips: 17.32890 ins/s
    2021-08-15 12:26:41,783 - INFO - epoch: 2, batch_id: 16, auc:0.976190,  avg_reader_cost: 0.00032 sec, avg_batch_cost: 0.11946 sec, avg_samples: 2.00000, ips: 16.74217 ins/s
    2021-08-15 12:26:42,027 - INFO - epoch: 2, batch_id: 18, auc:0.979167,  avg_reader_cost: 0.00031 sec, avg_batch_cost: 0.11933 sec, avg_samples: 2.00000, ips: 16.76077 ins/s
    2021-08-15 12:26:42,266 - INFO - epoch: 2, batch_id: 20, auc:0.979592,  avg_reader_cost: 0.00031 sec, avg_batch_cost: 0.11733 sec, avg_samples: 2.00000, ips: 17.04588 ins/s
    2021-08-15 12:26:42,506 - INFO - epoch: 2, batch_id: 22, auc:0.960961,  avg_reader_cost: 0.00032 sec, avg_batch_cost: 0.11762 sec, avg_samples: 2.00000, ips: 17.00379 ins/s
    2021-08-15 12:26:42,741 - INFO - epoch: 2, batch_id: 24, auc:0.967500,  avg_reader_cost: 0.00032 sec, avg_batch_cost: 0.11533 sec, avg_samples: 2.00000, ips: 17.34133 ins/s
    2021-08-15 12:26:42,976 - INFO - epoch: 2, batch_id: 26, auc:0.972516,  avg_reader_cost: 0.00032 sec, avg_batch_cost: 0.11530 sec, avg_samples: 2.00000, ips: 17.34671 ins/s
    2021-08-15 12:26:43,211 - INFO - epoch: 2, batch_id: 28, auc:0.975194,  avg_reader_cost: 0.00031 sec, avg_batch_cost: 0.11505 sec, avg_samples: 2.00000, ips: 17.38373 ins/s
    2021-08-15 12:26:43,446 - INFO - epoch: 2, batch_id: 30, auc:0.978261,  avg_reader_cost: 0.00034 sec, avg_batch_cost: 0.11551 sec, avg_samples: 2.00000, ips: 17.31381 ins/s
    2021-08-15 12:26:43,681 - INFO - epoch: 2, batch_id: 32, auc:0.980792,  avg_reader_cost: 0.00030 sec, avg_batch_cost: 0.11523 sec, avg_samples: 2.00000, ips: 17.35654 ins/s
    2021-08-15 12:26:43,918 - INFO - epoch: 2, batch_id: 34, auc:0.979592,  avg_reader_cost: 0.00032 sec, avg_batch_cost: 0.11614 sec, avg_samples: 2.00000, ips: 17.22015 ins/s
    2021-08-15 12:26:44,155 - INFO - epoch: 2, batch_id: 36, auc:0.981643,  avg_reader_cost: 0.00039 sec, avg_batch_cost: 0.11579 sec, avg_samples: 2.00000, ips: 17.27219 ins/s
    2021-08-15 12:26:44,388 - INFO - epoch: 2, batch_id: 38, auc:0.983399,  avg_reader_cost: 0.00032 sec, avg_batch_cost: 0.11447 sec, avg_samples: 2.00000, ips: 17.47172 ins/s
    2021-08-15 12:26:44,507 - INFO - epoch: 2 done, auc: 0.983982, epoch time: 4.74 s
    2021-08-15 12:26:44,871 - INFO - Already save model in output_model_dnn/2



```python
# 动态图预测
python -u PaddleRec/tools/infer.py -m PaddleRec/models/rank/dnn/config.yaml
```

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/setuptools/depends.py:2: DeprecationWarning: the imp module is deprecated in favour of importlib; see the module's documentation for alternative uses
      import imp
    2021-08-15 12:26:54,001 - INFO - **************common.configs**********
    2021-08-15 12:26:54,001 - INFO - use_gpu: False, use_visual: False, infer_batch_size: 2, test_data_dir: data/sample_data/train, start_epoch: 0, end_epoch: 3, print_interval: 2, model_load_path: output_model_dnn
    2021-08-15 12:26:54,001 - INFO - **************common.configs**********
    2021-08-15 12:26:54,123 - INFO - read data
    2021-08-15 12:26:54,123 - INFO - reader path:criteo_reader
    2021-08-15 12:26:54,124 - INFO - load model epoch 0
    2021-08-15 12:26:54,124 - INFO - start load model from output_model_dnn/0
    2021-08-15 12:26:54,356 - INFO - epoch: 0, batch_id: 0, auc: 1.000000, avg_reader_cost: 0.00118 sec, avg_batch_cost: 0.01605 sec, avg_samples: 2.00000, ips: 17.23 ins/s
    2021-08-15 12:26:54,372 - INFO - epoch: 0, batch_id: 2, auc: 1.000000, avg_reader_cost: 0.00026 sec, avg_batch_cost: 0.00622 sec, avg_samples: 2.00000, ips: 245.89 ins/s
    2021-08-15 12:26:54,388 - INFO - epoch: 0, batch_id: 4, auc: 0.812500, avg_reader_cost: 0.00019 sec, avg_batch_cost: 0.00576 sec, avg_samples: 2.00000, ips: 261.38 ins/s
    2021-08-15 12:26:54,403 - INFO - epoch: 0, batch_id: 6, auc: 0.909091, avg_reader_cost: 0.00019 sec, avg_batch_cost: 0.00549 sec, avg_samples: 2.00000, ips: 272.28 ins/s
    2021-08-15 12:26:54,418 - INFO - epoch: 0, batch_id: 8, auc: 0.911111, avg_reader_cost: 0.00018 sec, avg_batch_cost: 0.00544 sec, avg_samples: 2.00000, ips: 274.13 ins/s
    2021-08-15 12:26:54,433 - INFO - epoch: 0, batch_id: 10, auc: 0.930556, avg_reader_cost: 0.00018 sec, avg_batch_cost: 0.00569 sec, avg_samples: 2.00000, ips: 264.48 ins/s
    2021-08-15 12:26:54,448 - INFO - epoch: 0, batch_id: 12, auc: 0.904762, avg_reader_cost: 0.00018 sec, avg_batch_cost: 0.00581 sec, avg_samples: 2.00000, ips: 260.48 ins/s
    2021-08-15 12:26:54,463 - INFO - epoch: 0, batch_id: 14, auc: 0.888000, avg_reader_cost: 0.00021 sec, avg_batch_cost: 0.00550 sec, avg_samples: 2.00000, ips: 270.79 ins/s
    2021-08-15 12:26:54,478 - INFO - epoch: 0, batch_id: 16, auc: 0.904762, avg_reader_cost: 0.00018 sec, avg_batch_cost: 0.00548 sec, avg_samples: 2.00000, ips: 272.21 ins/s
    2021-08-15 12:26:54,493 - INFO - epoch: 0, batch_id: 18, auc: 0.885417, avg_reader_cost: 0.00020 sec, avg_batch_cost: 0.00562 sec, avg_samples: 2.00000, ips: 266.10 ins/s
    2021-08-15 12:26:54,508 - INFO - epoch: 0, batch_id: 20, auc: 0.906122, avg_reader_cost: 0.00020 sec, avg_batch_cost: 0.00548 sec, avg_samples: 2.00000, ips: 271.49 ins/s
    2021-08-15 12:26:54,523 - INFO - epoch: 0, batch_id: 22, auc: 0.861862, avg_reader_cost: 0.00020 sec, avg_batch_cost: 0.00548 sec, avg_samples: 2.00000, ips: 272.81 ins/s
    2021-08-15 12:26:54,538 - INFO - epoch: 0, batch_id: 24, auc: 0.885000, avg_reader_cost: 0.00020 sec, avg_batch_cost: 0.00565 sec, avg_samples: 2.00000, ips: 265.92 ins/s
    2021-08-15 12:26:54,553 - INFO - epoch: 0, batch_id: 26, auc: 0.890063, avg_reader_cost: 0.00019 sec, avg_batch_cost: 0.00567 sec, avg_samples: 2.00000, ips: 263.13 ins/s
    2021-08-15 12:26:54,568 - INFO - epoch: 0, batch_id: 28, auc: 0.903876, avg_reader_cost: 0.00021 sec, avg_batch_cost: 0.00544 sec, avg_samples: 2.00000, ips: 273.99 ins/s
    2021-08-15 12:26:54,583 - INFO - epoch: 0, batch_id: 30, auc: 0.915761, avg_reader_cost: 0.00029 sec, avg_batch_cost: 0.00555 sec, avg_samples: 2.00000, ips: 271.55 ins/s
    2021-08-15 12:26:54,598 - INFO - epoch: 0, batch_id: 32, auc: 0.923169, avg_reader_cost: 0.00020 sec, avg_batch_cost: 0.00553 sec, avg_samples: 2.00000, ips: 271.00 ins/s
    2021-08-15 12:26:54,612 - INFO - epoch: 0, batch_id: 34, auc: 0.910593, avg_reader_cost: 0.00018 sec, avg_batch_cost: 0.00550 sec, avg_samples: 2.00000, ips: 271.45 ins/s
    2021-08-15 12:26:54,625 - INFO - epoch: 0, batch_id: 36, auc: 0.919580, avg_reader_cost: 0.00019 sec, avg_batch_cost: 0.00434 sec, avg_samples: 2.00000, ips: 323.05 ins/s
    2021-08-15 12:26:54,637 - INFO - epoch: 0, batch_id: 38, auc: 0.927273, avg_reader_cost: 0.00017 sec, avg_batch_cost: 0.00414 sec, avg_samples: 2.00000, ips: 333.74 ins/s
    2021-08-15 12:26:54,645 - INFO - epoch: 0 done, auc: 0.929062, epoch time: 0.52 s
    2021-08-15 12:26:54,645 - INFO - load model epoch 1
    2021-08-15 12:26:54,645 - INFO - start load model from output_model_dnn/1
    2021-08-15 12:26:54,835 - INFO - epoch: 1, batch_id: 0, auc: 0.933190, avg_reader_cost: 0.00117 sec, avg_batch_cost: 0.00581 sec, avg_samples: 2.00000, ips: 20.21 ins/s
    2021-08-15 12:26:54,851 - INFO - epoch: 1, batch_id: 2, auc: 0.937500, avg_reader_cost: 0.00021 sec, avg_batch_cost: 0.00575 sec, avg_samples: 2.00000, ips: 261.88 ins/s
    2021-08-15 12:26:54,866 - INFO - epoch: 1, batch_id: 4, auc: 0.913846, avg_reader_cost: 0.00018 sec, avg_batch_cost: 0.00566 sec, avg_samples: 2.00000, ips: 263.93 ins/s
    2021-08-15 12:26:54,881 - INFO - epoch: 1, batch_id: 6, auc: 0.920814, avg_reader_cost: 0.00021 sec, avg_batch_cost: 0.00565 sec, avg_samples: 2.00000, ips: 267.22 ins/s
    2021-08-15 12:26:54,897 - INFO - epoch: 1, batch_id: 8, auc: 0.905716, avg_reader_cost: 0.00032 sec, avg_batch_cost: 0.00561 sec, avg_samples: 2.00000, ips: 256.27 ins/s
    2021-08-15 12:26:54,912 - INFO - epoch: 1, batch_id: 10, auc: 0.900988, avg_reader_cost: 0.00018 sec, avg_batch_cost: 0.00563 sec, avg_samples: 2.00000, ips: 266.03 ins/s
    2021-08-15 12:26:54,927 - INFO - epoch: 1, batch_id: 12, auc: 0.886218, avg_reader_cost: 0.00020 sec, avg_batch_cost: 0.00571 sec, avg_samples: 2.00000, ips: 262.83 ins/s
    2021-08-15 12:26:54,942 - INFO - epoch: 1, batch_id: 14, auc: 0.859974, avg_reader_cost: 0.00047 sec, avg_batch_cost: 0.00560 sec, avg_samples: 2.00000, ips: 268.36 ins/s
    2021-08-15 12:26:54,957 - INFO - epoch: 1, batch_id: 16, auc: 0.868763, avg_reader_cost: 0.00020 sec, avg_batch_cost: 0.00567 sec, avg_samples: 2.00000, ips: 264.53 ins/s
    2021-08-15 12:26:54,972 - INFO - epoch: 1, batch_id: 18, auc: 0.843278, avg_reader_cost: 0.00018 sec, avg_batch_cost: 0.00528 sec, avg_samples: 2.00000, ips: 280.89 ins/s
    2021-08-15 12:26:54,986 - INFO - epoch: 1, batch_id: 20, auc: 0.835326, avg_reader_cost: 0.00019 sec, avg_batch_cost: 0.00523 sec, avg_samples: 2.00000, ips: 280.43 ins/s
    2021-08-15 12:26:55,000 - INFO - epoch: 1, batch_id: 22, auc: 0.832945, avg_reader_cost: 0.00017 sec, avg_batch_cost: 0.00525 sec, avg_samples: 2.00000, ips: 281.99 ins/s
    2021-08-15 12:26:55,015 - INFO - epoch: 1, batch_id: 24, auc: 0.833958, avg_reader_cost: 0.00017 sec, avg_batch_cost: 0.00521 sec, avg_samples: 2.00000, ips: 281.43 ins/s
    2021-08-15 12:26:55,029 - INFO - epoch: 1, batch_id: 26, auc: 0.831324, avg_reader_cost: 0.00018 sec, avg_batch_cost: 0.00532 sec, avg_samples: 2.00000, ips: 277.77 ins/s
    2021-08-15 12:26:55,044 - INFO - epoch: 1, batch_id: 28, auc: 0.843816, avg_reader_cost: 0.00018 sec, avg_batch_cost: 0.00527 sec, avg_samples: 2.00000, ips: 280.98 ins/s
    2021-08-15 12:26:55,058 - INFO - epoch: 1, batch_id: 30, auc: 0.834578, avg_reader_cost: 0.00020 sec, avg_batch_cost: 0.00550 sec, avg_samples: 2.00000, ips: 269.93 ins/s
    2021-08-15 12:26:55,073 - INFO - epoch: 1, batch_id: 32, auc: 0.831722, avg_reader_cost: 0.00020 sec, avg_batch_cost: 0.00555 sec, avg_samples: 2.00000, ips: 270.64 ins/s
    2021-08-15 12:26:55,088 - INFO - epoch: 1, batch_id: 34, auc: 0.843375, avg_reader_cost: 0.00020 sec, avg_batch_cost: 0.00540 sec, avg_samples: 2.00000, ips: 276.47 ins/s
    2021-08-15 12:26:55,101 - INFO - epoch: 1, batch_id: 36, auc: 0.851070, avg_reader_cost: 0.00020 sec, avg_batch_cost: 0.00435 sec, avg_samples: 2.00000, ips: 320.37 ins/s
    2021-08-15 12:26:55,112 - INFO - epoch: 1, batch_id: 38, auc: 0.858210, avg_reader_cost: 0.00019 sec, avg_batch_cost: 0.00398 sec, avg_samples: 2.00000, ips: 343.67 ins/s
    2021-08-15 12:26:55,120 - INFO - epoch: 1 done, auc: 0.860507, epoch time: 0.47 s
    2021-08-15 12:26:55,120 - INFO - load model epoch 2
    2021-08-15 12:26:55,120 - INFO - start load model from output_model_dnn/2
    2021-08-15 12:26:55,307 - INFO - epoch: 2, batch_id: 0, auc: 0.864662, avg_reader_cost: 0.00108 sec, avg_batch_cost: 0.00565 sec, avg_samples: 2.00000, ips: 20.58 ins/s
    2021-08-15 12:26:55,322 - INFO - epoch: 2, batch_id: 2, auc: 0.869212, avg_reader_cost: 0.00021 sec, avg_batch_cost: 0.00576 sec, avg_samples: 2.00000, ips: 261.62 ins/s
    2021-08-15 12:26:55,338 - INFO - epoch: 2, batch_id: 4, auc: 0.874573, avg_reader_cost: 0.00021 sec, avg_batch_cost: 0.00588 sec, avg_samples: 2.00000, ips: 258.54 ins/s
    2021-08-15 12:26:55,353 - INFO - epoch: 2, batch_id: 6, auc: 0.880082, avg_reader_cost: 0.00019 sec, avg_batch_cost: 0.00562 sec, avg_samples: 2.00000, ips: 265.47 ins/s
    2021-08-15 12:26:55,368 - INFO - epoch: 2, batch_id: 8, auc: 0.883800, avg_reader_cost: 0.00022 sec, avg_batch_cost: 0.00556 sec, avg_samples: 2.00000, ips: 270.07 ins/s
    2021-08-15 12:26:55,383 - INFO - epoch: 2, batch_id: 10, auc: 0.888712, avg_reader_cost: 0.00020 sec, avg_batch_cost: 0.00553 sec, avg_samples: 2.00000, ips: 271.29 ins/s
    2021-08-15 12:26:55,398 - INFO - epoch: 2, batch_id: 12, auc: 0.893319, avg_reader_cost: 0.00019 sec, avg_batch_cost: 0.00547 sec, avg_samples: 2.00000, ips: 273.83 ins/s
    2021-08-15 12:26:55,412 - INFO - epoch: 2, batch_id: 14, auc: 0.896389, avg_reader_cost: 0.00021 sec, avg_batch_cost: 0.00538 sec, avg_samples: 2.00000, ips: 276.21 ins/s
    2021-08-15 12:26:55,427 - INFO - epoch: 2, batch_id: 16, auc: 0.900257, avg_reader_cost: 0.00018 sec, avg_batch_cost: 0.00524 sec, avg_samples: 2.00000, ips: 280.67 ins/s
    2021-08-15 12:26:55,441 - INFO - epoch: 2, batch_id: 18, auc: 0.902990, avg_reader_cost: 0.00020 sec, avg_batch_cost: 0.00539 sec, avg_samples: 2.00000, ips: 276.59 ins/s
    2021-08-15 12:26:55,457 - INFO - epoch: 2, batch_id: 20, auc: 0.906737, avg_reader_cost: 0.00020 sec, avg_batch_cost: 0.00570 sec, avg_samples: 2.00000, ips: 261.11 ins/s
    2021-08-15 12:26:55,472 - INFO - epoch: 2, batch_id: 22, auc: 0.909753, avg_reader_cost: 0.00020 sec, avg_batch_cost: 0.00564 sec, avg_samples: 2.00000, ips: 266.31 ins/s
    2021-08-15 12:26:55,487 - INFO - epoch: 2, batch_id: 24, auc: 0.913091, avg_reader_cost: 0.00020 sec, avg_batch_cost: 0.00556 sec, avg_samples: 2.00000, ips: 269.45 ins/s
    2021-08-15 12:26:55,502 - INFO - epoch: 2, batch_id: 26, auc: 0.916248, avg_reader_cost: 0.00023 sec, avg_batch_cost: 0.00549 sec, avg_samples: 2.00000, ips: 272.80 ins/s
    2021-08-15 12:26:55,516 - INFO - epoch: 2, batch_id: 28, auc: 0.921740, avg_reader_cost: 0.00020 sec, avg_batch_cost: 0.00545 sec, avg_samples: 2.00000, ips: 272.74 ins/s
    2021-08-15 12:26:55,537 - INFO - epoch: 2, batch_id: 30, auc: 0.924446, avg_reader_cost: 0.00022 sec, avg_batch_cost: 0.00657 sec, avg_samples: 2.00000, ips: 198.62 ins/s
    2021-08-15 12:26:55,560 - INFO - epoch: 2, batch_id: 32, auc: 0.927013, avg_reader_cost: 0.00029 sec, avg_batch_cost: 0.00821 sec, avg_samples: 2.00000, ips: 174.17 ins/s
    2021-08-15 12:26:55,580 - INFO - epoch: 2, batch_id: 34, auc: 0.931371, avg_reader_cost: 0.00028 sec, avg_batch_cost: 0.00815 sec, avg_samples: 2.00000, ips: 197.72 ins/s
    2021-08-15 12:26:55,593 - INFO - epoch: 2, batch_id: 36, auc: 0.933602, avg_reader_cost: 0.00022 sec, avg_batch_cost: 0.00435 sec, avg_samples: 2.00000, ips: 323.93 ins/s
    2021-08-15 12:26:55,605 - INFO - epoch: 2, batch_id: 38, auc: 0.935726, avg_reader_cost: 0.00018 sec, avg_batch_cost: 0.00397 sec, avg_samples: 2.00000, ips: 345.50 ins/s
    2021-08-15 12:26:55,613 - INFO - epoch: 2 done, auc: 0.936478, epoch time: 0.49 s



```python
# 静态图训练
python -u PaddleRec/tools/static_trainer.py -m PaddleRec/models/rank/dnn/config.yaml  # 全量数据运行config_bigdata.yaml 
```

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/setuptools/depends.py:2: DeprecationWarning: the imp module is deprecated in favour of importlib; see the module's documentation for alternative uses
      import imp
    2021-08-15 12:27:22,314 - INFO - cpu_num: 2
    2021-08-15 12:27:22,353 - INFO - **************common.configs**********
    2021-08-15 12:27:22,353 - INFO - use_gpu: False, use_visual: False, train_batch_size: 2, train_data_dir: data/sample_data/train, epochs: 3, print_interval: 2, model_save_path: output_model_dnn
    2021-08-15 12:27:22,353 - INFO - **************common.configs**********
    2021-08-15 12:27:22,522 - INFO - reader path:criteo_reader
    2021-08-15 12:27:22,523 - INFO - AUC Reset To Zero: _generated_var_0
    2021-08-15 12:27:22,525 - INFO - AUC Reset To Zero: _generated_var_1
    2021-08-15 12:27:22,525 - INFO - AUC Reset To Zero: _generated_var_2
    2021-08-15 12:27:22,525 - INFO - AUC Reset To Zero: _generated_var_3
    2021-08-15 12:27:22,572 - INFO - epoch: 0, batch_id: 0, cost: 0.71512246, auc: 0., avg_reader_cost: 0.00114 sec, avg_batch_cost: 0.02305 sec, avg_samples: 1.00000, ips: 43.37664 ins/s
    2021-08-15 12:27:22,612 - INFO - epoch: 0, batch_id: 2, cost: 0.650853, auc: 0.4, avg_reader_cost: 0.00014 sec, avg_batch_cost: 0.01975 sec, avg_samples: 2.00000, ips: 101.24994 ins/s
    2021-08-15 12:27:22,649 - INFO - epoch: 0, batch_id: 4, cost: 0.7863822, auc: 0.3125, avg_reader_cost: 0.00011 sec, avg_batch_cost: 0.01803 sec, avg_samples: 2.00000, ips: 110.95309 ins/s
    2021-08-15 12:27:22,693 - INFO - epoch: 0, batch_id: 6, cost: 0.30416656, auc: 0.36363636, avg_reader_cost: 0.00022 sec, avg_batch_cost: 0.02164 sec, avg_samples: 2.00000, ips: 92.40336 ins/s
    2021-08-15 12:27:22,728 - INFO - epoch: 0, batch_id: 8, cost: 0.22111034, auc: 0.53333333, avg_reader_cost: 0.00011 sec, avg_batch_cost: 0.01738 sec, avg_samples: 2.00000, ips: 115.06218 ins/s
    2021-08-15 12:27:22,767 - INFO - epoch: 0, batch_id: 10, cost: 0.15436125, auc: 0.45833333, avg_reader_cost: 0.00014 sec, avg_batch_cost: 0.01893 sec, avg_samples: 2.00000, ips: 105.64133 ins/s
    2021-08-15 12:27:22,806 - INFO - epoch: 0, batch_id: 12, cost: 0.12726486, auc: 0.44761905, avg_reader_cost: 0.00017 sec, avg_batch_cost: 0.01925 sec, avg_samples: 2.00000, ips: 103.89077 ins/s
    2021-08-15 12:27:22,854 - INFO - epoch: 0, batch_id: 14, cost: 0.1158806, auc: 0.512, avg_reader_cost: 0.00014 sec, avg_batch_cost: 0.02374 sec, avg_samples: 2.00000, ips: 84.24496 ins/s
    2021-08-15 12:27:22,891 - INFO - epoch: 0, batch_id: 16, cost: 0.0919114, auc: 0.4702381, avg_reader_cost: 0.00012 sec, avg_batch_cost: 0.01796 sec, avg_samples: 2.00000, ips: 111.34113 ins/s
    2021-08-15 12:27:22,927 - INFO - epoch: 0, batch_id: 18, cost: 0.10573484, auc: 0.515625, avg_reader_cost: 0.00015 sec, avg_batch_cost: 0.01763 sec, avg_samples: 2.00000, ips: 113.46306 ins/s
    2021-08-15 12:27:22,965 - INFO - epoch: 0, batch_id: 20, cost: 0.10134532, auc: 0.46530612, avg_reader_cost: 0.00012 sec, avg_batch_cost: 0.01877 sec, avg_samples: 2.00000, ips: 106.57546 ins/s
    2021-08-15 12:27:23,015 - INFO - epoch: 0, batch_id: 22, cost: 2.619127, auc: 0.38438438, avg_reader_cost: 0.00014 sec, avg_batch_cost: 0.02462 sec, avg_samples: 2.00000, ips: 81.23616 ins/s
    2021-08-15 12:27:23,053 - INFO - epoch: 0, batch_id: 24, cost: 0.19715421, auc: 0.3975, avg_reader_cost: 0.00012 sec, avg_batch_cost: 0.01895 sec, avg_samples: 2.00000, ips: 105.55027 ins/s
    2021-08-15 12:27:23,094 - INFO - epoch: 0, batch_id: 26, cost: 0.6882591, auc: 0.43763214, avg_reader_cost: 0.00014 sec, avg_batch_cost: 0.01967 sec, avg_samples: 2.00000, ips: 101.65361 ins/s
    2021-08-15 12:27:23,132 - INFO - epoch: 0, batch_id: 28, cost: 1.3056368, auc: 0.5255814, avg_reader_cost: 0.00017 sec, avg_batch_cost: 0.01906 sec, avg_samples: 2.00000, ips: 104.91137 ins/s
    2021-08-15 12:27:23,172 - INFO - epoch: 0, batch_id: 30, cost: 0.708484, auc: 0.52581522, avg_reader_cost: 0.00014 sec, avg_batch_cost: 0.01921 sec, avg_samples: 2.00000, ips: 104.12611 ins/s
    2021-08-15 12:27:23,264 - INFO - epoch: 0, batch_id: 32, cost: 0.6801218, auc: 0.51980792, avg_reader_cost: 0.00023 sec, avg_batch_cost: 0.04578 sec, avg_samples: 2.00000, ips: 43.69101 ins/s
    2021-08-15 12:27:23,304 - INFO - epoch: 0, batch_id: 34, cost: 0.95171344, auc: 0.58309038, avg_reader_cost: 0.00023 sec, avg_batch_cost: 0.01987 sec, avg_samples: 2.00000, ips: 100.63472 ins/s
    2021-08-15 12:27:23,338 - INFO - epoch: 0, batch_id: 36, cost: 0.6228216, auc: 0.5708042, avg_reader_cost: 0.00013 sec, avg_batch_cost: 0.01654 sec, avg_samples: 2.00000, ips: 120.90642 ins/s
    2021-08-15 12:27:23,372 - INFO - epoch: 0, batch_id: 38, cost: 0.7747818, auc: 0.56205534, avg_reader_cost: 0.00014 sec, avg_batch_cost: 0.01658 sec, avg_samples: 2.00000, ips: 120.61349 ins/s
    2021-08-15 12:27:23,389 - INFO - epoch: 0 done, cost: [0.76043403], auc: [0.5453852], epoch time: 0.87 s
    2021-08-15 12:27:23,790 - INFO - Already save model in output_model_dnn/0
    2021-08-15 12:27:23,790 - INFO - AUC Reset To Zero: _generated_var_0
    2021-08-15 12:27:23,791 - INFO - AUC Reset To Zero: _generated_var_1
    2021-08-15 12:27:23,791 - INFO - AUC Reset To Zero: _generated_var_2
    2021-08-15 12:27:23,791 - INFO - AUC Reset To Zero: _generated_var_3
    2021-08-15 12:27:23,817 - INFO - epoch: 1, batch_id: 0, cost: 0.5155326, auc: 1., avg_reader_cost: 0.00078 sec, avg_batch_cost: 0.01267 sec, avg_samples: 1.00000, ips: 78.94046 ins/s
    2021-08-15 12:27:23,856 - INFO - epoch: 1, batch_id: 2, cost: 0.64022124, auc: 1., avg_reader_cost: 0.00014 sec, avg_batch_cost: 0.01925 sec, avg_samples: 2.00000, ips: 103.91779 ins/s
    2021-08-15 12:27:23,897 - INFO - epoch: 1, batch_id: 4, cost: 0.7044856, auc: 0.8125, avg_reader_cost: 0.00015 sec, avg_batch_cost: 0.02014 sec, avg_samples: 2.00000, ips: 99.30696 ins/s
    2021-08-15 12:27:23,939 - INFO - epoch: 1, batch_id: 6, cost: 0.36067984, auc: 0.90909091, avg_reader_cost: 0.00017 sec, avg_batch_cost: 0.02054 sec, avg_samples: 2.00000, ips: 97.35517 ins/s
    2021-08-15 12:27:24,008 - INFO - epoch: 1, batch_id: 8, cost: 0.37630868, auc: 0.93333333, avg_reader_cost: 0.00017 sec, avg_batch_cost: 0.03390 sec, avg_samples: 2.00000, ips: 58.99784 ins/s
    2021-08-15 12:27:24,048 - INFO - epoch: 1, batch_id: 10, cost: 0.24587007, auc: 0.84722222, avg_reader_cost: 0.00015 sec, avg_batch_cost: 0.01973 sec, avg_samples: 2.00000, ips: 101.34841 ins/s
    2021-08-15 12:27:24,087 - INFO - epoch: 1, batch_id: 12, cost: 0.28718132, auc: 0.81904762, avg_reader_cost: 0.00012 sec, avg_batch_cost: 0.01924 sec, avg_samples: 2.00000, ips: 103.97125 ins/s
    2021-08-15 12:27:24,127 - INFO - epoch: 1, batch_id: 14, cost: 0.14732422, auc: 0.848, avg_reader_cost: 0.00017 sec, avg_batch_cost: 0.01962 sec, avg_samples: 2.00000, ips: 101.95507 ins/s
    2021-08-15 12:27:24,188 - INFO - epoch: 1, batch_id: 16, cost: 0.11715221, auc: 0.79761905, avg_reader_cost: 0.00016 sec, avg_batch_cost: 0.03000 sec, avg_samples: 2.00000, ips: 66.67441 ins/s
    2021-08-15 12:27:24,228 - INFO - epoch: 1, batch_id: 18, cost: 0.13614903, auc: 0.82291667, avg_reader_cost: 0.00018 sec, avg_batch_cost: 0.01967 sec, avg_samples: 2.00000, ips: 101.68133 ins/s
    2021-08-15 12:27:24,268 - INFO - epoch: 1, batch_id: 20, cost: 0.06046569, auc: 0.7755102, avg_reader_cost: 0.00014 sec, avg_batch_cost: 0.01947 sec, avg_samples: 2.00000, ips: 102.70402 ins/s
    2021-08-15 12:27:24,306 - INFO - epoch: 1, batch_id: 22, cost: 1.8510168, auc: 0.68468468, avg_reader_cost: 0.00013 sec, avg_batch_cost: 0.01879 sec, avg_samples: 2.00000, ips: 106.43146 ins/s
    2021-08-15 12:27:24,346 - INFO - epoch: 1, batch_id: 24, cost: 0.05389625, auc: 0.71, avg_reader_cost: 0.00014 sec, avg_batch_cost: 0.01973 sec, avg_samples: 2.00000, ips: 101.36066 ins/s
    2021-08-15 12:27:24,410 - INFO - epoch: 1, batch_id: 26, cost: 0.38552687, auc: 0.74841438, avg_reader_cost: 0.00023 sec, avg_batch_cost: 0.03145 sec, avg_samples: 2.00000, ips: 63.58474 ins/s
    2021-08-15 12:27:24,449 - INFO - epoch: 1, batch_id: 28, cost: 1.5468465, auc: 0.74263566, avg_reader_cost: 0.00016 sec, avg_batch_cost: 0.01917 sec, avg_samples: 2.00000, ips: 104.35150 ins/s
    2021-08-15 12:27:24,488 - INFO - epoch: 1, batch_id: 30, cost: 0.33722514, auc: 0.77377717, avg_reader_cost: 0.00018 sec, avg_batch_cost: 0.01906 sec, avg_samples: 2.00000, ips: 104.95074 ins/s
    2021-08-15 12:27:24,526 - INFO - epoch: 1, batch_id: 32, cost: 0.40158188, auc: 0.78931573, avg_reader_cost: 0.00014 sec, avg_batch_cost: 0.01877 sec, avg_samples: 2.00000, ips: 106.55786 ins/s
    2021-08-15 12:27:24,585 - INFO - epoch: 1, batch_id: 34, cost: 1.2027445, auc: 0.80563654, avg_reader_cost: 0.00024 sec, avg_batch_cost: 0.02920 sec, avg_samples: 2.00000, ips: 68.48151 ins/s
    2021-08-15 12:27:24,621 - INFO - epoch: 1, batch_id: 36, cost: 0.22388805, auc: 0.79195804, avg_reader_cost: 0.00014 sec, avg_batch_cost: 0.01780 sec, avg_samples: 2.00000, ips: 112.34617 ins/s
    2021-08-15 12:27:24,657 - INFO - epoch: 1, batch_id: 38, cost: 0.67811954, auc: 0.77628458, avg_reader_cost: 0.00012 sec, avg_batch_cost: 0.01743 sec, avg_samples: 2.00000, ips: 114.74582 ins/s
    2021-08-15 12:27:24,675 - INFO - epoch: 1 done, cost: [0.8691914], auc: [0.75819985], epoch time: 0.89 s
    2021-08-15 12:27:25,048 - INFO - Already save model in output_model_dnn/1
    2021-08-15 12:27:25,049 - INFO - AUC Reset To Zero: _generated_var_0
    2021-08-15 12:27:25,050 - INFO - AUC Reset To Zero: _generated_var_1
    2021-08-15 12:27:25,050 - INFO - AUC Reset To Zero: _generated_var_2
    2021-08-15 12:27:25,050 - INFO - AUC Reset To Zero: _generated_var_3
    2021-08-15 12:27:25,076 - INFO - epoch: 2, batch_id: 0, cost: 0.08261378, auc: 1., avg_reader_cost: 0.00089 sec, avg_batch_cost: 0.01272 sec, avg_samples: 1.00000, ips: 78.62600 ins/s
    2021-08-15 12:27:25,139 - INFO - epoch: 2, batch_id: 2, cost: 0.25904918, auc: 1., avg_reader_cost: 0.00023 sec, avg_batch_cost: 0.03148 sec, avg_samples: 2.00000, ips: 63.54140 ins/s
    2021-08-15 12:27:25,179 - INFO - epoch: 2, batch_id: 4, cost: 0.91894686, auc: 0.9375, avg_reader_cost: 0.00014 sec, avg_batch_cost: 0.01932 sec, avg_samples: 2.00000, ips: 103.50045 ins/s
    2021-08-15 12:27:25,216 - INFO - epoch: 2, batch_id: 6, cost: 0.02876424, auc: 0.96969697, avg_reader_cost: 0.00015 sec, avg_batch_cost: 0.01857 sec, avg_samples: 2.00000, ips: 107.70574 ins/s
    2021-08-15 12:27:25,254 - INFO - epoch: 2, batch_id: 8, cost: 0.11966871, auc: 0.97777778, avg_reader_cost: 0.00015 sec, avg_batch_cost: 0.01836 sec, avg_samples: 2.00000, ips: 108.93801 ins/s
    2021-08-15 12:27:25,316 - INFO - epoch: 2, batch_id: 10, cost: 0.02654589, auc: 0.97222222, avg_reader_cost: 0.00013 sec, avg_batch_cost: 0.03060 sec, avg_samples: 2.00000, ips: 65.35703 ins/s
    2021-08-15 12:27:25,355 - INFO - epoch: 2, batch_id: 12, cost: 0.06796515, auc: 0.97142857, avg_reader_cost: 0.00014 sec, avg_batch_cost: 0.01913 sec, avg_samples: 2.00000, ips: 104.54071 ins/s
    2021-08-15 12:27:25,393 - INFO - epoch: 2, batch_id: 14, cost: 0.01233004, auc: 0.976, avg_reader_cost: 0.00017 sec, avg_batch_cost: 0.01894 sec, avg_samples: 2.00000, ips: 105.59611 ins/s
    2021-08-15 12:27:25,432 - INFO - epoch: 2, batch_id: 16, cost: 0.01171245, auc: 0.97619048, avg_reader_cost: 0.00013 sec, avg_batch_cost: 0.01871 sec, avg_samples: 2.00000, ips: 106.90755 ins/s
    2021-08-15 12:27:25,493 - INFO - epoch: 2, batch_id: 18, cost: 0.02711548, auc: 0.97916667, avg_reader_cost: 0.00016 sec, avg_batch_cost: 0.03042 sec, avg_samples: 2.00000, ips: 65.75458 ins/s
    2021-08-15 12:27:25,533 - INFO - epoch: 2, batch_id: 20, cost: 0.01027189, auc: 0.97959184, avg_reader_cost: 0.00015 sec, avg_batch_cost: 0.01964 sec, avg_samples: 2.00000, ips: 101.83130 ins/s
    2021-08-15 12:27:25,572 - INFO - epoch: 2, batch_id: 22, cost: 1.0393912, auc: 0.96696697, avg_reader_cost: 0.00013 sec, avg_batch_cost: 0.01892 sec, avg_samples: 2.00000, ips: 105.68724 ins/s
    2021-08-15 12:27:25,611 - INFO - epoch: 2, batch_id: 24, cost: 0.00752806, auc: 0.9725, avg_reader_cost: 0.00017 sec, avg_batch_cost: 0.01905 sec, avg_samples: 2.00000, ips: 104.99869 ins/s
    2021-08-15 12:27:25,651 - INFO - epoch: 2, batch_id: 26, cost: 0.06589647, auc: 0.97674419, avg_reader_cost: 0.00016 sec, avg_batch_cost: 0.01963 sec, avg_samples: 2.00000, ips: 101.87273 ins/s
    2021-08-15 12:27:25,715 - INFO - epoch: 2, batch_id: 28, cost: 0.45915818, auc: 0.97984496, avg_reader_cost: 0.00027 sec, avg_batch_cost: 0.03166 sec, avg_samples: 2.00000, ips: 63.18031 ins/s
    2021-08-15 12:27:25,753 - INFO - epoch: 2, batch_id: 30, cost: 0.03761059, auc: 0.98233696, avg_reader_cost: 0.00030 sec, avg_batch_cost: 0.01891 sec, avg_samples: 2.00000, ips: 105.77720 ins/s
    2021-08-15 12:27:25,792 - INFO - epoch: 2, batch_id: 32, cost: 0.13791697, auc: 0.98439376, avg_reader_cost: 0.00015 sec, avg_batch_cost: 0.01916 sec, avg_samples: 2.00000, ips: 104.37097 ins/s
    2021-08-15 12:27:25,831 - INFO - epoch: 2, batch_id: 34, cost: 0.6634006, auc: 0.98542274, avg_reader_cost: 0.00014 sec, avg_batch_cost: 0.01888 sec, avg_samples: 2.00000, ips: 105.95088 ins/s
    2021-08-15 12:27:25,892 - INFO - epoch: 2, batch_id: 36, cost: 0.02350019, auc: 0.98688811, avg_reader_cost: 0.00012 sec, avg_batch_cost: 0.03022 sec, avg_samples: 2.00000, ips: 66.17396 ins/s
    2021-08-15 12:27:25,929 - INFO - epoch: 2, batch_id: 38, cost: 0.00675279, auc: 0.98814229, avg_reader_cost: 0.00016 sec, avg_batch_cost: 0.01804 sec, avg_samples: 2.00000, ips: 110.84020 ins/s
    2021-08-15 12:27:25,948 - INFO - epoch: 2 done, cost: [0.01657825], auc: [0.98855835], epoch time: 0.90 s
    2021-08-15 12:27:26,316 - INFO - Already save model in output_model_dnn/2



```python
# 静态图预测

python -u PaddleRec/tools/static_infer.py -m PaddleRec/models/rank/dnn/config.yaml
```

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/setuptools/depends.py:2: DeprecationWarning: the imp module is deprecated in favour of importlib; see the module's documentation for alternative uses
      import imp
    2021-08-15 12:27:48,416 - INFO - cpu_num: 2
    2021-08-15 12:27:48,416 - INFO - **************common.configs**********
    2021-08-15 12:27:48,416 - INFO - use_gpu: False, use_visual: False, infer_batch_size: 2, test_data_dir: data/sample_data/train, start_epoch: 0, end_epoch: 3, print_interval: 2, model_load_path: output_model_dnn
    2021-08-15 12:27:48,416 - INFO - **************common.configs**********
    2021-08-15 12:27:48,539 - INFO - reader path:criteo_reader
    2021-08-15 12:27:48,540 - INFO - load model epoch 0
    2021-08-15 12:27:48,540 - INFO - start load model from output_model_dnn/0
    2021-08-15 12:27:48,773 - INFO - AUC Reset To Zero: _generated_var_0
    2021-08-15 12:27:48,774 - INFO - AUC Reset To Zero: _generated_var_1
    2021-08-15 12:27:48,774 - INFO - AUC Reset To Zero: _generated_var_2
    2021-08-15 12:27:48,774 - INFO - AUC Reset To Zero: _generated_var_3
    2021-08-15 12:27:48,811 - INFO - epoch: 0, batch_id: 0, auc: 1.0, avg_reader_cost: 0.00189 sec, avg_batch_cost: 0.01881 sec, avg_samples: 2.00000, ips: 106.18 ins/s
    2021-08-15 12:27:48,833 - INFO - epoch: 0, batch_id: 2, auc: 1.0, avg_reader_cost: 0.00013 sec, avg_batch_cost: 0.01128 sec, avg_samples: 2.00000, ips: 177.06 ins/s
    2021-08-15 12:27:48,855 - INFO - epoch: 0, batch_id: 4, auc: 0.875, avg_reader_cost: 0.00012 sec, avg_batch_cost: 0.01076 sec, avg_samples: 2.00000, ips: 185.67 ins/s
    2021-08-15 12:27:48,877 - INFO - epoch: 0, batch_id: 6, auc: 0.9393939393939394, avg_reader_cost: 0.00013 sec, avg_batch_cost: 0.01098 sec, avg_samples: 2.00000, ips: 181.89 ins/s
    2021-08-15 12:27:48,900 - INFO - epoch: 0, batch_id: 8, auc: 0.9333333333333333, avg_reader_cost: 0.00013 sec, avg_batch_cost: 0.01096 sec, avg_samples: 2.00000, ips: 182.26 ins/s
    2021-08-15 12:27:48,922 - INFO - epoch: 0, batch_id: 10, auc: 0.9444444444444444, avg_reader_cost: 0.00013 sec, avg_batch_cost: 0.01082 sec, avg_samples: 2.00000, ips: 184.55 ins/s
    2021-08-15 12:27:48,944 - INFO - epoch: 0, batch_id: 12, auc: 0.9333333333333333, avg_reader_cost: 0.00014 sec, avg_batch_cost: 0.01089 sec, avg_samples: 2.00000, ips: 183.40 ins/s
    2021-08-15 12:27:48,965 - INFO - epoch: 0, batch_id: 14, auc: 0.92, avg_reader_cost: 0.00013 sec, avg_batch_cost: 0.01079 sec, avg_samples: 2.00000, ips: 185.22 ins/s
    2021-08-15 12:27:48,994 - INFO - epoch: 0, batch_id: 16, auc: 0.9285714285714286, avg_reader_cost: 0.00014 sec, avg_batch_cost: 0.01416 sec, avg_samples: 2.00000, ips: 141.05 ins/s
    2021-08-15 12:27:49,016 - INFO - epoch: 0, batch_id: 18, auc: 0.9166666666666666, avg_reader_cost: 0.00011 sec, avg_batch_cost: 0.01076 sec, avg_samples: 2.00000, ips: 185.60 ins/s
    2021-08-15 12:27:49,037 - INFO - epoch: 0, batch_id: 20, auc: 0.9306122448979591, avg_reader_cost: 0.00012 sec, avg_batch_cost: 0.01071 sec, avg_samples: 2.00000, ips: 186.53 ins/s
    2021-08-15 12:27:49,059 - INFO - epoch: 0, batch_id: 22, auc: 0.8858858858858859, avg_reader_cost: 0.00012 sec, avg_batch_cost: 0.01083 sec, avg_samples: 2.00000, ips: 184.43 ins/s
    2021-08-15 12:27:49,082 - INFO - epoch: 0, batch_id: 24, auc: 0.905, avg_reader_cost: 0.00013 sec, avg_batch_cost: 0.01091 sec, avg_samples: 2.00000, ips: 183.05 ins/s
    2021-08-15 12:27:49,104 - INFO - epoch: 0, batch_id: 26, auc: 0.9112050739957717, avg_reader_cost: 0.00012 sec, avg_batch_cost: 0.01091 sec, avg_samples: 2.00000, ips: 182.99 ins/s
    2021-08-15 12:27:49,125 - INFO - epoch: 0, batch_id: 28, auc: 0.9271317829457364, avg_reader_cost: 0.00012 sec, avg_batch_cost: 0.01065 sec, avg_samples: 2.00000, ips: 187.52 ins/s
    2021-08-15 12:27:49,147 - INFO - epoch: 0, batch_id: 30, auc: 0.936141304347826, avg_reader_cost: 0.00012 sec, avg_batch_cost: 0.01065 sec, avg_samples: 2.00000, ips: 187.53 ins/s
    2021-08-15 12:27:49,170 - INFO - epoch: 0, batch_id: 32, auc: 0.9411764705882353, avg_reader_cost: 0.00013 sec, avg_batch_cost: 0.01127 sec, avg_samples: 2.00000, ips: 177.20 ins/s
    2021-08-15 12:27:49,203 - INFO - epoch: 0, batch_id: 34, auc: 0.9251700680272109, avg_reader_cost: 0.00015 sec, avg_batch_cost: 0.01626 sec, avg_samples: 2.00000, ips: 122.90 ins/s
    2021-08-15 12:27:49,222 - INFO - epoch: 0, batch_id: 36, auc: 0.9326923076923077, avg_reader_cost: 0.00013 sec, avg_batch_cost: 0.00981 sec, avg_samples: 2.00000, ips: 203.45 ins/s
    2021-08-15 12:27:49,242 - INFO - epoch: 0, batch_id: 38, auc: 0.9391304347826087, avg_reader_cost: 0.00012 sec, avg_batch_cost: 0.00945 sec, avg_samples: 2.00000, ips: 211.30 ins/s
    2021-08-15 12:27:49,252 - INFO - epoch: 0 done, auc: 0.9405034324942791, epoch time: 0.48 s
    2021-08-15 12:27:49,252 - INFO - load model epoch 1
    2021-08-15 12:27:49,252 - INFO - start load model from output_model_dnn/1
    2021-08-15 12:27:49,483 - INFO - AUC Reset To Zero: _generated_var_0
    2021-08-15 12:27:49,483 - INFO - AUC Reset To Zero: _generated_var_1
    2021-08-15 12:27:49,483 - INFO - AUC Reset To Zero: _generated_var_2
    2021-08-15 12:27:49,483 - INFO - AUC Reset To Zero: _generated_var_3
    2021-08-15 12:27:49,501 - INFO - epoch: 1, batch_id: 0, auc: 1.0, avg_reader_cost: 0.00160 sec, avg_batch_cost: 0.00943 sec, avg_samples: 2.00000, ips: 211.75 ins/s
    2021-08-15 12:27:49,522 - INFO - epoch: 1, batch_id: 2, auc: 1.0, avg_reader_cost: 0.00011 sec, avg_batch_cost: 0.01057 sec, avg_samples: 2.00000, ips: 189.00 ins/s
    2021-08-15 12:27:49,544 - INFO - epoch: 1, batch_id: 4, auc: 0.9375, avg_reader_cost: 0.00012 sec, avg_batch_cost: 0.01049 sec, avg_samples: 2.00000, ips: 190.47 ins/s
    2021-08-15 12:27:49,565 - INFO - epoch: 1, batch_id: 6, auc: 0.9696969696969696, avg_reader_cost: 0.00012 sec, avg_batch_cost: 0.01052 sec, avg_samples: 2.00000, ips: 189.87 ins/s
    2021-08-15 12:27:49,586 - INFO - epoch: 1, batch_id: 8, auc: 0.9555555555555556, avg_reader_cost: 0.00012 sec, avg_batch_cost: 0.01034 sec, avg_samples: 2.00000, ips: 193.24 ins/s
    2021-08-15 12:27:49,607 - INFO - epoch: 1, batch_id: 10, auc: 0.9583333333333334, avg_reader_cost: 0.00012 sec, avg_batch_cost: 0.01059 sec, avg_samples: 2.00000, ips: 188.60 ins/s
    2021-08-15 12:27:49,636 - INFO - epoch: 1, batch_id: 12, auc: 0.9428571428571428, avg_reader_cost: 0.00012 sec, avg_batch_cost: 0.01422 sec, avg_samples: 2.00000, ips: 140.45 ins/s
    2021-08-15 12:27:49,658 - INFO - epoch: 1, batch_id: 14, auc: 0.9279999999999999, avg_reader_cost: 0.00012 sec, avg_batch_cost: 0.01074 sec, avg_samples: 2.00000, ips: 185.95 ins/s
    2021-08-15 12:27:49,679 - INFO - epoch: 1, batch_id: 16, auc: 0.9285714285714286, avg_reader_cost: 0.00012 sec, avg_batch_cost: 0.01058 sec, avg_samples: 2.00000, ips: 188.86 ins/s
    2021-08-15 12:27:49,700 - INFO - epoch: 1, batch_id: 18, auc: 0.9114583333333334, avg_reader_cost: 0.00011 sec, avg_batch_cost: 0.01045 sec, avg_samples: 2.00000, ips: 191.21 ins/s
    2021-08-15 12:27:49,721 - INFO - epoch: 1, batch_id: 20, auc: 0.9142857142857143, avg_reader_cost: 0.00012 sec, avg_batch_cost: 0.01037 sec, avg_samples: 2.00000, ips: 192.64 ins/s
    2021-08-15 12:27:49,742 - INFO - epoch: 1, batch_id: 22, auc: 0.8768768768768769, avg_reader_cost: 0.00011 sec, avg_batch_cost: 0.01044 sec, avg_samples: 2.00000, ips: 191.41 ins/s
    2021-08-15 12:27:49,764 - INFO - epoch: 1, batch_id: 24, auc: 0.8975, avg_reader_cost: 0.00012 sec, avg_batch_cost: 0.01051 sec, avg_samples: 2.00000, ips: 190.07 ins/s
    2021-08-15 12:27:49,785 - INFO - epoch: 1, batch_id: 26, auc: 0.9006342494714588, avg_reader_cost: 0.00011 sec, avg_batch_cost: 0.01034 sec, avg_samples: 2.00000, ips: 193.24 ins/s
    2021-08-15 12:27:49,806 - INFO - epoch: 1, batch_id: 28, auc: 0.9085271317829459, avg_reader_cost: 0.00011 sec, avg_batch_cost: 0.01045 sec, avg_samples: 2.00000, ips: 191.10 ins/s
    2021-08-15 12:27:49,834 - INFO - epoch: 1, batch_id: 30, auc: 0.9198369565217391, avg_reader_cost: 0.00012 sec, avg_batch_cost: 0.01400 sec, avg_samples: 2.00000, ips: 142.73 ins/s
    2021-08-15 12:27:49,891 - INFO - epoch: 1, batch_id: 32, auc: 0.9267707082833133, avg_reader_cost: 0.00012 sec, avg_batch_cost: 0.02840 sec, avg_samples: 2.00000, ips: 70.39 ins/s
    2021-08-15 12:27:49,912 - INFO - epoch: 1, batch_id: 34, auc: 0.924198250728863, avg_reader_cost: 0.00011 sec, avg_batch_cost: 0.01055 sec, avg_samples: 2.00000, ips: 189.31 ins/s
    2021-08-15 12:27:49,932 - INFO - epoch: 1, batch_id: 36, auc: 0.9318181818181818, avg_reader_cost: 0.00012 sec, avg_batch_cost: 0.00945 sec, avg_samples: 2.00000, ips: 211.33 ins/s
    2021-08-15 12:27:49,951 - INFO - epoch: 1, batch_id: 38, auc: 0.9383399209486166, avg_reader_cost: 0.00011 sec, avg_batch_cost: 0.00947 sec, avg_samples: 2.00000, ips: 211.00 ins/s
    2021-08-15 12:27:49,961 - INFO - epoch: 1 done, auc: 0.9405034324942791, epoch time: 0.48 s
    2021-08-15 12:27:49,961 - INFO - load model epoch 2
    2021-08-15 12:27:49,961 - INFO - start load model from output_model_dnn/2
    2021-08-15 12:27:50,195 - INFO - AUC Reset To Zero: _generated_var_0
    2021-08-15 12:27:50,196 - INFO - AUC Reset To Zero: _generated_var_1
    2021-08-15 12:27:50,196 - INFO - AUC Reset To Zero: _generated_var_2
    2021-08-15 12:27:50,196 - INFO - AUC Reset To Zero: _generated_var_3
    2021-08-15 12:27:50,213 - INFO - epoch: 2, batch_id: 0, auc: 1.0, avg_reader_cost: 0.00197 sec, avg_batch_cost: 0.00913 sec, avg_samples: 2.00000, ips: 218.71 ins/s
    2021-08-15 12:27:50,235 - INFO - epoch: 2, batch_id: 2, auc: 1.0, avg_reader_cost: 0.00013 sec, avg_batch_cost: 0.01091 sec, avg_samples: 2.00000, ips: 183.01 ins/s
    2021-08-15 12:27:50,256 - INFO - epoch: 2, batch_id: 4, auc: 1.0, avg_reader_cost: 0.00012 sec, avg_batch_cost: 0.01042 sec, avg_samples: 2.00000, ips: 191.76 ins/s
    2021-08-15 12:27:50,278 - INFO - epoch: 2, batch_id: 6, auc: 1.0, avg_reader_cost: 0.00012 sec, avg_batch_cost: 0.01050 sec, avg_samples: 2.00000, ips: 190.31 ins/s
    2021-08-15 12:27:50,310 - INFO - epoch: 2, batch_id: 8, auc: 1.0, avg_reader_cost: 0.00012 sec, avg_batch_cost: 0.01597 sec, avg_samples: 2.00000, ips: 125.09 ins/s
    2021-08-15 12:27:50,335 - INFO - epoch: 2, batch_id: 10, auc: 1.0, avg_reader_cost: 0.00015 sec, avg_batch_cost: 0.01231 sec, avg_samples: 2.00000, ips: 162.27 ins/s
    2021-08-15 12:27:50,360 - INFO - epoch: 2, batch_id: 12, auc: 1.0, avg_reader_cost: 0.00015 sec, avg_batch_cost: 0.01254 sec, avg_samples: 2.00000, ips: 159.21 ins/s
    2021-08-15 12:27:50,385 - INFO - epoch: 2, batch_id: 14, auc: 1.0, avg_reader_cost: 0.00018 sec, avg_batch_cost: 0.01211 sec, avg_samples: 2.00000, ips: 164.89 ins/s
    2021-08-15 12:27:50,409 - INFO - epoch: 2, batch_id: 16, auc: 0.994047619047619, avg_reader_cost: 0.00015 sec, avg_batch_cost: 0.01196 sec, avg_samples: 2.00000, ips: 167.03 ins/s
    2021-08-15 12:27:50,434 - INFO - epoch: 2, batch_id: 18, auc: 0.984375, avg_reader_cost: 0.00018 sec, avg_batch_cost: 0.01204 sec, avg_samples: 2.00000, ips: 165.89 ins/s
    2021-08-15 12:27:50,456 - INFO - epoch: 2, batch_id: 20, auc: 0.9877551020408163, avg_reader_cost: 0.00013 sec, avg_batch_cost: 0.01097 sec, avg_samples: 2.00000, ips: 182.01 ins/s
    2021-08-15 12:27:50,478 - INFO - epoch: 2, batch_id: 22, auc: 0.981981981981982, avg_reader_cost: 0.00012 sec, avg_batch_cost: 0.01088 sec, avg_samples: 2.00000, ips: 183.56 ins/s
    2021-08-15 12:27:50,500 - INFO - epoch: 2, batch_id: 24, auc: 0.985, avg_reader_cost: 0.00013 sec, avg_batch_cost: 0.01107 sec, avg_samples: 2.00000, ips: 180.31 ins/s
    2021-08-15 12:27:50,523 - INFO - epoch: 2, batch_id: 26, auc: 0.9873150105708245, avg_reader_cost: 0.00014 sec, avg_batch_cost: 0.01120 sec, avg_samples: 2.00000, ips: 178.36 ins/s
    2021-08-15 12:27:50,555 - INFO - epoch: 2, batch_id: 28, auc: 0.9906976744186047, avg_reader_cost: 0.00018 sec, avg_batch_cost: 0.01582 sec, avg_samples: 2.00000, ips: 126.26 ins/s
    2021-08-15 12:27:50,578 - INFO - epoch: 2, batch_id: 30, auc: 0.9918478260869565, avg_reader_cost: 0.00015 sec, avg_batch_cost: 0.01125 sec, avg_samples: 2.00000, ips: 177.44 ins/s
    2021-08-15 12:27:50,600 - INFO - epoch: 2, batch_id: 32, auc: 0.9927971188475391, avg_reader_cost: 0.00013 sec, avg_batch_cost: 0.01103 sec, avg_samples: 2.00000, ips: 181.11 ins/s
    2021-08-15 12:27:50,623 - INFO - epoch: 2, batch_id: 34, auc: 0.9941690962099126, avg_reader_cost: 0.00013 sec, avg_batch_cost: 0.01096 sec, avg_samples: 2.00000, ips: 182.25 ins/s
    2021-08-15 12:27:50,642 - INFO - epoch: 2, batch_id: 36, auc: 0.9947552447552448, avg_reader_cost: 0.00013 sec, avg_batch_cost: 0.00982 sec, avg_samples: 2.00000, ips: 203.24 ins/s
    2021-08-15 12:27:50,662 - INFO - epoch: 2, batch_id: 38, auc: 0.9952569169960475, avg_reader_cost: 0.00013 sec, avg_batch_cost: 0.00961 sec, avg_samples: 2.00000, ips: 207.74 ins/s
    2021-08-15 12:27:50,672 - INFO - epoch: 2 done, auc: 0.9954233409610984, epoch time: 0.48 s


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

    --2021-08-15 12:16:18--  https://paddlerec.bj.bcebos.com/datasets/criteo/slot_test_data_full.tar.gz
    Resolving paddlerec.bj.bcebos.com (paddlerec.bj.bcebos.com)... 182.61.200.229, 182.61.200.195, 2409:8c00:6c21:10ad:0:ff:b00e:67d, ...
    Connecting to paddlerec.bj.bcebos.com (paddlerec.bj.bcebos.com)|182.61.200.229|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 171502914 (164M) [application/x-gzip]
    Saving to: ‘slot_test_data_full.tar.gz’
    
    slot_test_data_full 100%[===================>] 163.56M  49.2MB/s    in 4.0s    
    
    2021-08-15 12:16:23 (40.7 MB/s) - ‘slot_test_data_full.tar.gz’ saved [171502914/171502914]
    
    slot_test_data_full/
    slot_test_data_full/part-223
    slot_test_data_full/part-224
    slot_test_data_full/part-221
    slot_test_data_full/part-222
    slot_test_data_full/part-226
    slot_test_data_full/part-225
    slot_test_data_full/part-228
    slot_test_data_full/part-229
    slot_test_data_full/part-227
    slot_test_data_full/part-220
    --2021-08-15 12:16:28--  https://paddlerec.bj.bcebos.com/datasets/criteo/slot_train_data_full.tar.gz
    Resolving paddlerec.bj.bcebos.com (paddlerec.bj.bcebos.com)... 182.61.200.229, 182.61.200.195, 2409:8c04:1001:1002:0:ff:b001:368a, ...
    Connecting to paddlerec.bj.bcebos.com (paddlerec.bj.bcebos.com)|182.61.200.229|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 4083608889 (3.8G) [application/x-gzip]
    Saving to: ‘slot_train_data_full.tar.gz’
    
    slot_train_data_ful 100%[===================>]   3.80G  17.6MB/s    in 3m 7s   
    
    2021-08-15 12:19:35 (20.8 MB/s) - ‘slot_train_data_full.tar.gz’ saved [4083608889/4083608889]
    
    slot_train_data_full/
    slot_train_data_full/part-170
    slot_train_data_full/part-45
    slot_train_data_full/part-152
    slot_train_data_full/part-80
    slot_train_data_full/part-14
    slot_train_data_full/part-166
    slot_train_data_full/part-192
    slot_train_data_full/part-148
    slot_train_data_full/part-46
    slot_train_data_full/part-87
    slot_train_data_full/part-188
    slot_train_data_full/part-165
    slot_train_data_full/part-8
    slot_train_data_full/part-115
    slot_train_data_full/part-189
    slot_train_data_full/part-2
    slot_train_data_full/part-47
    slot_train_data_full/part-174
    slot_train_data_full/part-215
    slot_train_data_full/part-63
    slot_train_data_full/part-163
    slot_train_data_full/part-183
    slot_train_data_full/part-12
    slot_train_data_full/part-51
    slot_train_data_full/part-178
    slot_train_data_full/part-65
    slot_train_data_full/part-172
    slot_train_data_full/part-98
    slot_train_data_full/part-9
    slot_train_data_full/part-92
    slot_train_data_full/part-207
    slot_train_data_full/part-197
    slot_train_data_full/part-206
    slot_train_data_full/part-111
    slot_train_data_full/part-187
    slot_train_data_full/part-15
    slot_train_data_full/part-44
    slot_train_data_full/part-161
    slot_train_data_full/part-19
    slot_train_data_full/part-127
    slot_train_data_full/part-134
    slot_train_data_full/part-101
    slot_train_data_full/part-27
    slot_train_data_full/part-120
    slot_train_data_full/part-97
    slot_train_data_full/part-79
    slot_train_data_full/part-162
    slot_train_data_full/part-20
    slot_train_data_full/part-26
    slot_train_data_full/part-138
    slot_train_data_full/part-86
    slot_train_data_full/part-153
    slot_train_data_full/part-126
    slot_train_data_full/part-149
    slot_train_data_full/part-213
    slot_train_data_full/part-159
    slot_train_data_full/part-210
    slot_train_data_full/part-69
    slot_train_data_full/part-205
    slot_train_data_full/part-0
    slot_train_data_full/part-164
    slot_train_data_full/part-39
    slot_train_data_full/part-110
    slot_train_data_full/part-64
    slot_train_data_full/part-193
    slot_train_data_full/part-141
    slot_train_data_full/part-107
    slot_train_data_full/part-145
    slot_train_data_full/part-125
    slot_train_data_full/part-124
    slot_train_data_full/part-25
    slot_train_data_full/part-42
    slot_train_data_full/part-81
    slot_train_data_full/part-1
    slot_train_data_full/part-128
    slot_train_data_full/part-179
    slot_train_data_full/part-66
    slot_train_data_full/part-34
    slot_train_data_full/part-117
    slot_train_data_full/part-121
    slot_train_data_full/part-147
    slot_train_data_full/part-133
    slot_train_data_full/part-155
    slot_train_data_full/part-6
    slot_train_data_full/part-28
    slot_train_data_full/part-4
    slot_train_data_full/part-106
    slot_train_data_full/part-137
    slot_train_data_full/part-55
    slot_train_data_full/part-78
    slot_train_data_full/part-180
    slot_train_data_full/part-94
    slot_train_data_full/part-136
    slot_train_data_full/part-90
    slot_train_data_full/part-93
    slot_train_data_full/part-146
    slot_train_data_full/part-104
    slot_train_data_full/part-218
    slot_train_data_full/part-22
    slot_train_data_full/part-77
    slot_train_data_full/part-67
    slot_train_data_full/part-114
    slot_train_data_full/part-49
    slot_train_data_full/part-36
    slot_train_data_full/part-74
    slot_train_data_full/part-209
    slot_train_data_full/part-190
    slot_train_data_full/part-37
    slot_train_data_full/part-194
    slot_train_data_full/part-196
    slot_train_data_full/part-13
    slot_train_data_full/part-95
    slot_train_data_full/part-84
    slot_train_data_full/part-118
    slot_train_data_full/part-144
    slot_train_data_full/part-112
    slot_train_data_full/part-156
    slot_train_data_full/part-105
    slot_train_data_full/part-32
    slot_train_data_full/part-108
    slot_train_data_full/part-41
    slot_train_data_full/part-143
    slot_train_data_full/part-139
    slot_train_data_full/part-198
    slot_train_data_full/part-214
    slot_train_data_full/part-85
    slot_train_data_full/part-203
    slot_train_data_full/part-88
    slot_train_data_full/part-56
    slot_train_data_full/part-176
    slot_train_data_full/part-123
    slot_train_data_full/part-202
    slot_train_data_full/part-50
    slot_train_data_full/part-91
    slot_train_data_full/part-59
    slot_train_data_full/part-73
    slot_train_data_full/part-168
    slot_train_data_full/part-191
    slot_train_data_full/part-212
    slot_train_data_full/part-23
    slot_train_data_full/part-103
    slot_train_data_full/part-195
    slot_train_data_full/part-61
    slot_train_data_full/part-182
    slot_train_data_full/part-135
    slot_train_data_full/part-208
    slot_train_data_full/part-70
    slot_train_data_full/part-217
    slot_train_data_full/part-60
    slot_train_data_full/part-142
    slot_train_data_full/part-169
    slot_train_data_full/part-99
    slot_train_data_full/part-216
    slot_train_data_full/part-16
    slot_train_data_full/part-199
    slot_train_data_full/part-72
    slot_train_data_full/part-184
    slot_train_data_full/part-173
    slot_train_data_full/part-68
    slot_train_data_full/part-40
    slot_train_data_full/part-82
    slot_train_data_full/part-43
    slot_train_data_full/part-53
    slot_train_data_full/part-89
    slot_train_data_full/part-58
    slot_train_data_full/part-219
    slot_train_data_full/part-151
    slot_train_data_full/part-201
    slot_train_data_full/part-200
    slot_train_data_full/part-21
    slot_train_data_full/part-83
    slot_train_data_full/part-33
    slot_train_data_full/part-181
    slot_train_data_full/part-186
    slot_train_data_full/part-62
    slot_train_data_full/part-129
    slot_train_data_full/part-113
    slot_train_data_full/part-38
    slot_train_data_full/part-122
    slot_train_data_full/part-31
    slot_train_data_full/part-17
    slot_train_data_full/part-109
    slot_train_data_full/part-54
    slot_train_data_full/part-11
    slot_train_data_full/part-116
    slot_train_data_full/part-96
    slot_train_data_full/part-10
    slot_train_data_full/part-7
    slot_train_data_full/part-76
    slot_train_data_full/part-211
    slot_train_data_full/part-18
    slot_train_data_full/part-102
    slot_train_data_full/part-140
    slot_train_data_full/part-35
    slot_train_data_full/part-29
    slot_train_data_full/part-132
    slot_train_data_full/part-150
    slot_train_data_full/part-204
    slot_train_data_full/part-119
    slot_train_data_full/part-185
    slot_train_data_full/part-52
    slot_train_data_full/part-177
    slot_train_data_full/part-100
    slot_train_data_full/part-5
    slot_train_data_full/part-75
    slot_train_data_full/part-131
    slot_train_data_full/part-3
    slot_train_data_full/part-157
    slot_train_data_full/part-24
    slot_train_data_full/part-154
    slot_train_data_full/part-171
    slot_train_data_full/part-167
    slot_train_data_full/part-57
    slot_train_data_full/part-71
    slot_train_data_full/part-175
    slot_train_data_full/part-158
    slot_train_data_full/part-30
    slot_train_data_full/part-130
    slot_train_data_full/part-160



```python
#查看config配置文件

cat PaddleRec/models/rank/dnn/config_bigdata.yaml
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

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/setuptools/depends.py:2: DeprecationWarning: the imp module is deprecated in favour of importlib; see the module's documentation for alternative uses
      import imp
    2021-08-15 13:37:13,337 - INFO - **************common.configs**********
    2021-08-15 13:37:13,338 - INFO - use_gpu: False, use_visual: False, train_batch_size: 512, train_data_dir: ../../../datasets/criteo/slot_train_data_full, epochs: 4, print_interval: 10, model_save_path: output_model_dnn_all
    2021-08-15 13:37:13,338 - INFO - **************common.configs**********
    2021-08-15 13:37:13,461 - INFO - read data
    2021-08-15 13:37:13,462 - INFO - reader path:criteo_reader
    2021-08-15 13:37:13,819 - INFO - epoch: 0, batch_id: 0, auc:0.479725,  avg_reader_cost: 0.00698 sec, avg_batch_cost: 0.03524 sec, avg_samples: 51.20000, ips: 1452.81652 ins/s
    2021-08-15 13:37:15,756 - INFO - epoch: 0, batch_id: 10, auc:0.523818,  avg_reader_cost: 0.00034 sec, avg_batch_cost: 0.19312 sec, avg_samples: 512.00000, ips: 2651.26937 ins/s
    2021-08-15 13:37:17,542 - INFO - epoch: 0, batch_id: 20, auc:0.573902,  avg_reader_cost: 0.00033 sec, avg_batch_cost: 0.17797 sec, avg_samples: 512.00000, ips: 2876.93709 ins/s
    2021-08-15 13:37:19,439 - INFO - epoch: 0, batch_id: 30, auc:0.600810,  avg_reader_cost: 0.00041 sec, avg_batch_cost: 0.18908 sec, avg_samples: 512.00000, ips: 2707.89463 ins/s
    2021-08-15 13:37:21,300 - INFO - epoch: 0, batch_id: 40, auc:0.616033,  avg_reader_cost: 0.00032 sec, avg_batch_cost: 0.18551 sec, avg_samples: 512.00000, ips: 2759.95454 ins/s
    2021-08-15 13:37:23,109 - INFO - epoch: 0, batch_id: 50, auc:0.628529,  avg_reader_cost: 0.00032 sec, avg_batch_cost: 0.18037 sec, avg_samples: 512.00000, ips: 2838.64559 ins/s
    2021-08-15 13:37:25,000 - INFO - epoch: 0, batch_id: 60, auc:0.641285,  avg_reader_cost: 0.00033 sec, avg_batch_cost: 0.18847 sec, avg_samples: 512.00000, ips: 2716.61126 ins/s
    2021-08-15 13:37:26,842 - INFO - epoch: 0, batch_id: 70, auc:0.650621,  avg_reader_cost: 0.00034 sec, avg_batch_cost: 0.18355 sec, avg_samples: 512.00000, ips: 2789.39879 ins/s
    2021-08-15 13:37:28,581 - INFO - epoch: 0, batch_id: 80, auc:0.656475,  avg_reader_cost: 0.00036 sec, avg_batch_cost: 0.17328 sec, avg_samples: 512.00000, ips: 2954.67236 ins/s
    2021-08-15 13:37:30,357 - INFO - epoch: 0, batch_id: 90, auc:0.662073,  avg_reader_cost: 0.00033 sec, avg_batch_cost: 0.17700 sec, avg_samples: 512.00000, ips: 2892.73372 ins/s
    2021-08-15 13:37:32,092 - INFO - epoch: 0, batch_id: 100, auc:0.665430,  avg_reader_cost: 0.00035 sec, avg_batch_cost: 0.17290 sec, avg_samples: 512.00000, ips: 2961.25933 ins/s
    2021-08-15 13:37:33,865 - INFO - epoch: 0, batch_id: 110, auc:0.669695,  avg_reader_cost: 0.00035 sec, avg_batch_cost: 0.17665 sec, avg_samples: 512.00000, ips: 2898.41599 ins/s



```python
# 动态图预测
python -u PaddleRec/tools/infer.py -m PaddleRec/models/rank/dnn/config_bigdata.yaml  # 全量数据运行config_bigdata.yaml 
```

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/setuptools/depends.py:2: DeprecationWarning: the imp module is deprecated in favour of importlib; see the module's documentation for alternative uses
      import imp
    2021-08-15 13:29:33,558 - INFO - **************common.configs**********
    2021-08-15 13:29:33,559 - INFO - use_gpu: False, use_visual: False, infer_batch_size: 512, test_data_dir: ../../../datasets/criteo/slot_test_data_full, start_epoch: 0, end_epoch: 4, print_interval: 10, model_load_path: output_model_dnn_all
    2021-08-15 13:29:33,559 - INFO - **************common.configs**********
    2021-08-15 13:29:33,683 - INFO - read data
    2021-08-15 13:29:33,683 - INFO - reader path:criteo_reader
    2021-08-15 13:29:33,684 - INFO - load model epoch 0
    2021-08-15 13:29:33,684 - INFO - start load model from output_model_dnn_all/0
    Traceback (most recent call last):
      File "PaddleRec/tools/infer.py", line 187, in <module>
        main(args)
      File "PaddleRec/tools/infer.py", line 115, in main
        load_model(model_path, dy_model)
      File "/home/aistudio/PaddleRec/tools/utils/save_load.py", line 44, in load_model
        param_state_dict = paddle.load(model_prefix + ".pdparams")
      File "/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/framework/io.py", line 905, in load
        load_result = _legacy_load(path, **configs)
      File "/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/framework/io.py", line 924, in _legacy_load
        model_path, config = _build_load_path_and_config(path, config)
      File "/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/framework/io.py", line 161, in _build_load_path_and_config
        raise ValueError(error_msg % path)
    ValueError: The ``path`` (output_model_dnn_all/0/rec.pdparams) to load model not exists.


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
