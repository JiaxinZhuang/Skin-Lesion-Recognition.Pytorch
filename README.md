# README
## About Data
All data has been downloaded including from task 1 to task 3 , and all of them 
are put in Lab-DataServer 

**/home/share2/MIA/Dataset/ISIC-2018/**. 

![Data Directory](./Data_Directory.png)


## Directory
```
main directory
.
├── Data_Directory.png
├── README.md
├── data
└── src
    ├── data_utils.py
    └── requirement.txt

data (You have to copy dataset into data directory)
├── ISIC2018_Task1-2_Training_Input
├── ISIC2018_Task1_Training_GroundTruth
├── ISIC2018_Task2_Training_GroundTruth_v3
├── ISIC2018_Task3_Training_GroundTruth
└── ISIC2018_Task3_Training_Input
```

## Different file functionality
```
src/train train and evaluate
src/model wrap different model to use
src/data_utils generate train, valid data and load data
src/resNet
src/vgg19
src/train_xgboost.py

```

## Using Jihan's method to process data
```
python data_utils.py --ISIC2018_Task3_Training_Input=/home/share2/MIA/ISIC2018-Sharing/jihan/ggw_p2s3
python data_utils.py --ISIC2018_Task3_Training_Input=/home/share2/MIA/ISIC2018-Sharing/jihan/gw
python data_utils.py --ISIC2018_Task3_Training_Input=/home/share2/MIA/ISIC2018-Sharing/jihan/sog6
python data_utils.py --ISIC2018_Task3_Training_Input=/home/share2/MIA/ISIC2018-Sharing/jihan/wpr1
```
output\_file will generate in src/task3\_32\_300\_400

```
mv task3\_32\_300\_400/\* /home/jiaxin/jihan/20180606/Reverse\_CISI\_Classification/data/ISIC2018/2018\_6\_4/task3\_32\_300\_400 
```

# Run
```
python train.py --remove=True --CUDA_VISIBLE_DEVICE=7
```

# logs models saved
logs, models, tra and val wuold be saved in ../save\_32\_300\_400

# Some ways to solve imbalanced data
1. median class weight
```
weight_sample_ = np.array([1113,6705,514,327,1099,115,142])/10015
weight_sample_ = 0.05132302/weight_sample_
```

## Class weight


## Until 7-9
### Code directory 

```
src_sy
├── main.py
├── models
│   ├── CNN_Train.py
│   ├── DatasetFolder.py
│   ├── FineTune.py
│   ├── FocalLoss.py
│   ├── ReadCSV.py
│   ├── __init__.py
│   └── focalloss2d.py
└── tags
```

### Train & Evaluate
```
python main.py --c CUDA_VISIBLE_DEVICE --logfile=FILENAME
```

## Ref
1. [Tensorflow — Dealing with imbalanced data](https://blog.node.us.com/tensorflow-dealing-with-imbalanced-data-eb0108b10701)
2. [xgboost](https://xgboost.readthedocs.io/en/latest/get_started/index.html)
