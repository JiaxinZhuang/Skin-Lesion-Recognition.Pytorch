import numpy as np
import xgboost as xgb
from sklearn.decomposition import PCA
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score
from xgboost import plot_importance
import matplotlib
matplotlib.use('Agg') # solved problem: _tkinter.TclError: no display name and no $DISPLAY environment variable
from matplotlib import pyplot as plt
def batch_flatten(batch_arr):
    batch_arr_flatten = []
    for i in range(batch_arr.shape[0]):
        for j in range(batch_arr[i].shape[0]):
            batch_arr_flatten.append(batch_arr[i][j])

    batch_arr_flatten = np.array(batch_arr_flatten)

    return batch_arr_flatten
if __name__ == '__main__':

    features_path = "/home/jiaxin/myGithub/Reverse_CISI_Classification/src/"
    # features_train = np.load(features_path + "images_feature_with_labels_from_vgg16_train.npy")
    data_train = np.load(features_path + "images_feature_with_labels_from_vgg16_train.npy")
    features_train = batch_flatten(data_train[0])

    print(features_train.shape)
    print(features_train[0])
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(features_train)
    features_train = scaler.transform(features_train)
    print(features_train[0])
    # label_train = batch_flatten(np.reshape(data_train[1], (-1, 1)))
    label_train = batch_flatten(data_train[1])
    print(label_train)
    from sklearn.utils import shuffle
    features_train, label_train = shuffle(features_train, label_train, random_state=13)
    print(label_train)
    train_time_start = time.time()
    xgc = xgb.XGBClassifier(base_score=0.6, colsample_bylevel=0.7, colsample_bytree=1,
           gamma=0, learning_rate=0.1, max_delta_step=3, max_depth=6,
           min_child_weight=1, missing=None, num_class=7, n_estimators=150, nthread=-1,
           # objective='binary:logistic', reg_alpha=0, reg_lambda=1,
           # objective='rank:pairwise', reg_alpha=0, reg_lambda=1,
           objective='multi:softmax', reg_alpha=0, reg_lambda=1,
           scale_pos_weight=1, seed=49, eval_metric='merror', silent=True, subsample=0.8)
    xgc.fit(features_train, label_train)
    train_time_end = time.time();
    print('Time of training is: ' + str(train_time_end - train_time_start) + ' s.')

    data_test = np.load(features_path + "images_feature_with_labels_from_vgg16_evaluate.npy")
    features_test = batch_flatten(data_test[0])
    # label_test = batch_flatten(np.reshape(data_test[1], (-1, 1)))
    label_test = batch_flatten(data_test[1])

    features_test = scaler.transform(features_test)

    pred_test = xgc.predict(features_test)
    np.savetxt('xgboost.csv', pred_test, delimiter = ',')
    predictions_test = [round(value) for value in pred_test]
    accuracy_test = accuracy_score(label_test, predictions_test)
    print("XGBoost Testing Accuracy: %.2f%%" % (accuracy_test * 100.0))
    pred_train = xgc.predict(features_train)
    np.savetxt('xgboost_tr.csv', pred_train, delimiter = ',')
    predictions_train = [round(value) for value in pred_train]
    accuracy_train = accuracy_score(label_train, predictions_train)
    print("XGBoost Training Accuracy: %.2f%%" % (accuracy_train * 100.0))
    # plot_importance(xgc)
    # plt.savefig("importance_xgb_smote3.png")
    # plt.close()
