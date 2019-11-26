# README

New!!: Code has been updated with very basic settings. [2019/11/26]

New!!: Code and README would be updated very soon [2019/11/19]



## Directory

* src: contains all source codes
* scripts: bash scripts to train the model under different settings
* data: all images and csv file for splitting all data into training set and validation set.

```bash
. 
├── README.md
├── data
├── scripts
├── src
└── tags
```

# Some ways to solve imbalanced data
1. median class weight
```
weight_sample_ = np.array([1113,6705,514,327,1099,115,142])/10015
weight_sample_ = 0.05132302/weight_sample_
```

2. Class weight

### Train & Evaluate

See the scripts file.

### tensorboard

mca, lr 

```
tensorboard --logdir run
```

<img src="/Users/lincolnzjx/Desktop/ISIC_2018_Classification/001.png" style="zoom:33%;" />