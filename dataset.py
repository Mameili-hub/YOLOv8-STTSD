#作者CSDN主页：https://blog.csdn.net/m0_64879847
#YOLO目标检测数据集大全：https://blog.csdn.net/m0_64879847/article/details/132301975
import os
import random

# 划分比例
trainval_percent = 0.3
train_percent = 0.7

# 图片路径
images_path = r"/root/autodl-fs/ultralytics/datasets/images"
# txt写入地址
txtsavepath = images_path.replace('\images','\ImageSets\Main')
os.makedirs(txtsavepath, exist_ok=True)
total_img = os.listdir(images_path)

num = len(total_img)
list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)


ftrainval = open(txtsavepath+'\\trainval_list.txt', 'w')
ftest = open(txtsavepath+'\\test_list.txt', 'w')
ftrain = open(txtsavepath+'\\train_list.txt', 'w')
fval = open(txtsavepath+'\\val_list.txt', 'w')
for i in list:
    name = total_img[i] + '\n'
    img_path = 'datasets/images/' + name
    if i in trainval:
        ftrainval.write(img_path)
        if i in train:
            ftest.write(img_path)
        else:
            fval.write(img_path)
    else:
        ftrain.write(img_path)

ftrainval.close()
ftrain.close()
fval.close()
ftest.close()