import os
import random

val_percent = 0.05
test_percent = 0.05
train_percent = 0.9
img_path = '../dataset/preimages'

total_img = os.listdir(img_path)
num = len(total_img)
lst = range(num)
nv = int(num * val_percent)
nt = int(num * train_percent)

train_lst = random.sample(lst, nt)
val_test_lst = [x for x in lst if x not in train_lst]
val_lst = random.sample(val_test_lst, nv)
test_lst = [x for x in val_test_lst if x not in val_lst]

with open('../dataset/test.txt', 'w+') as ftest, open('../dataset/train.txt',
                                                      'w+') as ftrain, open('../dataset/val.txt', 'w+') as fval:
    for i in lst:
        name = total_img[i][:-4] + '\n'
        if i in train_lst:
            ftrain.write(name)
        elif i in val_lst:
            fval.write(name)
        elif i in test_lst:
            ftest.write(name)
