#!/usr/env/python python3
# -*- coding: utf-8 -*-
# @File     : tes.py
# @Time     : 2019/3/14 10:20 
# @Software : PyCharm
import os
import pickle
import random
from collections import defaultdict
import pandas as pd
from os.path import isfile

import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm


DATA_PATH = "./data/origin"
list_data = pd.read_csv("./data/list.csv")
list_df = pd.DataFrame(list_data)

# 得到训练集所有id
ids = set(list_df.loc[:, 'id'].values.tolist())

if isfile("./data/id2img.pickle"):
    with open("./data/id2img.pickle", "rb") as f:
        id2img = pickle.load(f)
else:
    id2img = defaultdict()
    for id in ids:
        id2img[id] = [os.path.join(DATA_PATH, id[:2], id+'_'+p+".jpg") for p in 'abc']
    f = open("./data/id2img.pickle", "wb")
    pickle.dump(id2img, f)

if isfile("./data/id2attr.pickle"):
    with open("./data/id2attr.pickle", "rb") as f:
        id2attr = pickle.load(f)
else:
    id2attr = defaultdict()
    for _, row in list_df.iterrows():
        id, x, y, judge = row
        id2attr[id] = (x,y,judge)
    f = open("./data/id2attr.pickle", "wb")
    pickle.dump(id2attr, f)

with open("./data/id2img.pickle", "rb") as f:
    id2img = pickle.load(f)

with open("./data/id2attr.pickle", "rb") as f:
    id2attr = pickle.load(f)

star = ["newtarget", "isstar", "asteroid", "isnova", "known"]
nonstar = ["noise", "ghost", "pity"]

name_list = star + nonstar
count_dict = dict(zip(name_list, [0]*8))

def plot_distribution(save_path):

    for key, val in id2attr.items():
        count_dict[val[2]]+=1
    print(count_dict)
    num_list = list(count_dict.values())

    plt.bar(range(len(num_list)), num_list,color='rgb',tick_label=name_list)
    plt.show()
    # plt.savefig(save_path)

# 绘制类别分布图
# plot_distribution("./distribution.jpg")


def split_train_validate(partition=0.2):
    """
    分割训练集和测试集
    :param partition:
    :return: train_dict, validate_dict  type->list
    """
    label2img = defaultdict(list)
    for key, val in id2img.items():
        label2img[id2attr[key][-1]].append(tuple(val))
    train_dict = {}
    validate_dict = {}

    for key, val in label2img.items():
        val_count = int(len(val)*partition)
        val_list = random.sample(val, val_count)
        validate_dict[key] = val_list
        for v in val_list:
            val.remove(v)
        train_dict[key] = val
    return train_dict, validate_dict

train_dict, validate_dict = split_train_validate()

def merge_img(data_dict, save_dir="./data", category="train"):
    record_list = []
    for cat, img_list in tqdm(data_dict.items()):
        for img_tuple in img_list:
            dif, new, old = img_tuple
            img1,_,_ = cv2.split(cv2.imread(dif))
            img2,_,_ = cv2.split(cv2.imread(new))
            img3,_,_ = cv2.split(cv2.imread(old))
            # # 合并三个通道
            merge = cv2.merge([img1, img2, img3])

            id = dif[17:len(dif)-6]
            dir = os.path.join(save_dir,"merge", category, id[:2])
            if os.path.exists(dir) is False:
                os.makedirs(dir)
            save_path = os.path.join(dir, id+".jpg")
            cv2.imwrite(save_path, merge)

            # cv2.imshow("merge", merge)
            # cv2.waitKey(0)
            record_list.append((save_path, id))

    cat2label = {**dict(zip(star, ["1"] * 5)), **dict(zip(nonstar, ["0"] * 3))}
    with open(os.path.join(save_dir, category+".txt"), "wb+") as f:
        for record in record_list:
            path, id = record
            center_x, center_y, cat = id2attr[id]
            boundary_box = [center_x-7, center_y-7, center_x+7, center_y+7]
            boundary_box = map(lambda x:str(x), boundary_box)
            line = path + " " + cat2label[cat]+ " " + " ".join(boundary_box) + "\n"
            if category=="train" and cat in star:
                # 将变星样本重复10遍
                for _ in range(10):
                    f.write(str(line).encode("utf-8"))
            else:
                f.write(str(line).encode("utf-8"))


# merge_img(validate_dict, category="val")
merge_img(train_dict, category="train")