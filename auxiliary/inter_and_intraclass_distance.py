# encoding:utf-8
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def max_intraclass_and_min_interclass_samples(features, labels, ind):
	''' 
	Input: 所有图片的特征：二维的list，
		   所有图片的类标：一维的list，
		   基准图片的序号：int,
	Output: 图片序号三元组：(基准图片序号，最大类内距离图片的序号，最小类间距离图片的序号)，(int, int, int)
			最大类内距离，double
			最小类间距离, double'''
	dists = np.sqrt(np.sum(np.square(np.array(features) - np.array(features[ind])), axis = 1))
	# find the sample with max intraclass distance
	intrac_inds = [x for x in range(len(dists)) if (labels[x] == labels[ind]) and (x != ind)]
	max_intrac_dis = dists[intrac_inds[0]]
	max_intrac_ind = intrac_inds[0]
	for i in intrac_inds:
		if dists[i] > max_intrac_dis:
			max_intrac_dis = dists[i]
			max_intrac_ind = i
	# find the sample with min interclass distance
	interc_inds = [x for x in range(len(dists)) if labels[x] != labels[ind]]
	min_interc_dis = dists[interc_inds[0]]
	min_interc_ind = interc_inds[0]
	for i in interc_inds:
		if dists[i] < min_interc_dis:
			min_interc_dis = dists[i]
			min_interc_ind = i
	return (ind, max_intrac_ind, min_interc_ind), max_intrac_dis, min_interc_dis

# 使用示例:max_intraclass_and_min_interclass_samples函数
features = [[1,2],[3,4],[4,5],[6,7],[1,4],[0,2],[4,4],[2,5]]
labels = [1,2,2,1,2,2,1,2]
ind = 2

(ind, max_intrac_ind, min_interc_ind),_,_ = max_intraclass_and_min_interclass_samples(features, labels, ind)

plt.figure()
plt.scatter(x=np.array(features)[:,0], y=np.array(features)[:,1], c=labels)
plt.scatter(x=features[max_intrac_ind][0],y=features[max_intrac_ind][1],c='green',marker='*',label='Max Intra-class Sample')
plt.scatter(x=features[min_interc_ind][0],y=features[min_interc_ind][1],c='red',marker='*',label='Min Inter-class Sample')
plt.scatter(x=features[ind][0],y=features[ind][1],c='blue',marker='*',label='Benchmark Sample')
plt.legend()
plt.title("An Example")
plt.show()

def show_img_triplets(img_root_dir, img_file_names, triplets_of_img_ind, labels):
	''' 
	Input: 图片的文件夹路径：string，
		   所有图片的文件名（相对路径）：string，
		   图片序号三元组(基准图片序号，最大类内距离图片的序号，最小类间距离图片的序号)：(int, int, int)
		   所有图片的类标：一维的list
		   基准图片的序号：int,
	Output: 图片序号三元组：(基准图片序号，最大类内距离图片的序号，最小类间距离图片的序号)，(int, int, int)
			最大类内距离，double
			最小类间距离, double'''
	(ind, max_intrac_ind, min_interc_ind) = triplets_of_img_ind
	img = Image.open(os.path.join(img_root_dir, img_file_names[ind]))
	img_max_intra = Image.open(os.path.join(img_root_dir, img_file_names[max_intrac_ind]))
	img_min_inter = Image.open(os.path.join(img_root_dir, img_file_names[min_interc_ind]))
	plt.figure()
	plt.subplot(1,3,1)
	plt.title("Image:"+img_file_names[ind] + "\nLable:" + str(labels[ind]))
	plt.imshow(img)
	plt.subplot(1,3,2)
	plt.title("Max Intra-class Image:" + img_file_names[max_intrac_ind] \
		+ "\nLable:" + str(labels[max_intrac_ind]))
	plt.imshow(img_max_intra)
	plt.subplot(1,3,3)
	plt.title("Min Inter-class Image:" + img_file_names[min_interc_ind] \
		+ "\nLable:" + str(labels[min_interc_ind]))
	plt.imshow(img_min_inter)
	plt.show()
# 使用示例:max_intraclass_and_min_interclass_samples函数+show_img_triplets函数
triplets_of_img_ind, max_intrac_dis, min_interc_dis = max_intraclass_and_min_interclass_samples(features, labels, ind)
show_img_triplets('H:\TMP\imgs\Men10m', \
	['010668.png', '010670.png', '010672.png', '010674.png', '010676.png', '010678.png', '010680.png', '010682.png'], \
	triplets_of_img_ind, labels)


