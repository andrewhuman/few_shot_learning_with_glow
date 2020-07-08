
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import numpy as np

def normal_img(img):
    return ((img / 255.0) - 0.5)*2

def expend_HWC(img):
    return np.expand_dims(img,axis=0)

def to_CHW(img):
    return np.transpose(img,(0,3,1,2))

def to_HWC(img):
    return np.transpose(img,(0,2,3,1))

def get_image_paths_and_labels(dataset):
    image_paths_flat = []
    labels_flat = []
    for i in range(len(dataset)):
        image_paths_flat += dataset[i].image_paths
        labels_flat += [i] * len(dataset[i].image_paths)
    return image_paths_flat, labels_flat

def get_image_paths(facedir):
    image_paths = []
    if os.path.isdir(facedir):
        images = os.listdir(facedir)
        images
        image_paths = [os.path.join(facedir,img) for img in images]
    return image_paths

def get_images_frompath(facedir):

    if os.path.isdir(facedir):
        images_l = os.listdir(facedir)
        images_l.sort()
        image_paths = [os.path.join(facedir,img) for img in images_l]
        # print(len(image_paths))
        images = []
        for image_p in image_paths:
            # print(image_p)
            # img_o = cv2.imread(image_p, 0)
            #  images.append(img_o)
            img_r = cv2.resize(cv2.imread(image_p, 0), (32, 32), interpolation=cv2.INTER_LINEAR)
            if img_r is not None:
                # print(img_r.shape)
                img_r = normal_img(img_r)
                img_r = np.expand_dims(img_r, axis=0)
                images.append(img_r)
            else:print("imread error")
        # print(len(images))
    return images

def get_images_frompath_random(facedir,seed=66):

    if os.path.isdir(facedir):
        images_l = os.listdir(facedir)
        image_paths = [os.path.join(facedir,img) for img in images_l]
        # print(image_paths)
        np.random.seed(seed)
        np.random.shuffle(image_paths)
        print(image_paths)
        images = []
        for image_p in image_paths:
            img_r = cv2.resize(cv2.imread(image_p, 0), (32, 32), interpolation=cv2.INTER_LINEAR)
            if img_r is not None:
                img_r = normal_img(img_r)
                img_r = np.expand_dims(img_r, axis=0)
                # print(img_r.shape)
                images.append(img_r)
            else:print("imread error")
    return images

def get_images_frompath_random_list_sort(facedir,seed=66,resize_mode = cv2.INTER_LINEAR):

    if os.path.isdir(facedir):
        images_l = os.listdir(facedir)
        images_l.sort()
        image_paths = [os.path.join(facedir,img) for img in images_l]
        # print(image_paths)
        np.random.seed(seed)
        np.random.shuffle(image_paths)
        # print(image_paths)
        images = []
        for image_p in image_paths:
            img_r = cv2.resize(cv2.imread(image_p, 0), (32, 32), interpolation=resize_mode)
            if img_r is not None:
                img_r = normal_img(img_r)
                img_r = np.expand_dims(img_r, axis=0)
                # print(img_r.shape)
                images.append(img_r)
            else:print("imread error")
    return images


def get_images_frompath_random_list_sort_wh(facedir,w,h,limit,seed=66):

    if os.path.isdir(facedir):
        images = os.listdir(facedir)
        images.sort()
        image_paths = [os.path.join(facedir,img) for img in images]
        # print(image_paths)
        np.random.seed(seed)
        np.random.shuffle(image_paths)
        image_paths=image_paths[:limit]
        # print(image_paths)
        images = []
        for image_p in image_paths:
            img_r = cv2.resize(cv2.imread(image_p, cv2.IMREAD_COLOR), (w, h), interpolation=cv2.INTER_LINEAR)
            if img_r is not None:
                img_r = normal_img(img_r)
                # img_r = np.expand_dims(img_r, axis=0)
                # print(img_r.shape)
                img_r = np.transpose(img_r,(2,0,1))
                # print(img_r.shape)
                images.append(img_r)
            else:print("imread error")
    return images

def get_images_from_images_list(image_paths):

    print(image_paths)
    images = []
    for image_p in image_paths:
        img_r = cv2.resize(cv2.imread(image_p, 0), (32, 32), interpolation=cv2.INTER_LINEAR)
        if img_r is not None:
            img_r = normal_img(img_r)
            img_r = np.expand_dims(img_r, axis=0)
            # print(img_r.shape)
            images.append(img_r)
        else:
            print("imread error")


    return images



class ImageClass():
    "Stores the paths to images for a given class"

    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths

    def setImgs(self,imgs):
        self.imgs = imgs

    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'

    def __len__(self):
        return len(self.image_paths)

class MeanVarObject():

    def __init__(self,label):
        self.label = label
        self.list=[]

    def add_z(self,z):
        ' it is well if z is numpy '
        self.list.append(z)

    def get_z_array(self):
        return np.array(self.list)

    def get_mean(self):
        return np.mean(self.get_z_array(),axis=0)

    def get_std(self):
        return np.std(self.get_z_array(),axis=0)

    def get_mean_std(self):
        return self.get_mean(),self.get_std()






class ImageObject():
    "Stores the paths to images for a given class"

    def __init__(self, name, imgs):
        self.name = name
        self.imgs = imgs

    def setImgs(self,imgs):
        self.imgs = imgs

    def set_label(self,label):
        self.label = label

    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'

    def __len__(self):
        return len(self.image_paths)


def get_dataset(path, has_class_directories=True):
    dataset = []
    path_exp = os.path.expanduser(path)
    classes = [path for path in os.listdir(path_exp) \
               if os.path.isdir(os.path.join(path_exp, path))]
    classes.sort()
    nrof_classes = len(classes)
    for i in range(nrof_classes):
        class_name = classes[i]
        print('class_name = ',class_name)
        facedir = os.path.join(path_exp, class_name)
        image_paths = get_image_paths(facedir)
        # print('image_paths = ',image_paths)
        dataset.append(ImageClass(class_name, image_paths))

    return dataset


def main():

    # negative
    # path_total = 'D:\Downloads\omniglot-master\python\images_background_small1'
    #
    # dataset_negative = []
    # for path_p in os.listdir(path_total):
    #     path_p_join = os.path.join(path_total,path_p)
    #     for path_c in os.listdir(path_p_join):
    #         path_real = os.path.join(path_p_join,path_c)
    #         class_name = path_real
    #         # image_paths = get_image_paths(path_real)
    #         images = get_images_frompath(path_real)
    #         print(class_name," : ",len(images))
    #         dataset_negative.append(ImageObject(class_name,images))
    #
    # print(len(dataset))
    # for img_object in dataset:
    #     print( len(img_object.imgs))

    # positive
    path_positive_total = 'D:\Downloads\omniglot-master\python\images_evaluation'
    class_name_list = []
    for path_p in os.listdir(path_positive_total):
        path_p_join = os.path.join(path_positive_total,path_p)
        for path_c in os.listdir(path_p_join):
            path_real = os.path.join(path_p_join,path_c)
            class_name = path_real
            class_name_list.append(str(class_name))

    # class_name = np.array(class_name)
    # print(class_name_list)
    np.random.seed(66)
    shufindex = np.random.permutation(len(class_name_list))
    # print(shufindex)
    class_positive = np.array(class_name_list)[shufindex][:5]
    print(class_positive)

    dataset_train_positive =[]
    dataset_test_positive = []
    for class_path in class_positive :
        images = get_images_frompath_random(class_path) # random is guding
        print(class_path, " : ", len(images))
        dataset_train_positive.append(ImageObject(class_path, images[:5]))
        dataset_test_positive.append(ImageObject(class_path, images[5:]))


    # label
    # print(len(dataset_train_positive[2].imgs))
    # print(len(dataset_test_positive[2].imgs))
    # class_total_num = len(dataset_negative) + len(dataset_train_positive)
    # print("class_total = ",class_total_num)
    # np.random.seed(66)
    # shufindex = np.random.permutation(class_total_num)
    # for index, value in enumerate(dataset_negative):
    #     value.label = shufindex[index]
    #     print(value.name,  value.label)
    #
    # for index, value in enumerate(dataset_train_positive):
    #     value.label = shufindex[index+len(dataset_negative)]
    #     dataset_test_positive[index].label = shufindex[index+len(dataset_negative)]
    #     print(value.name, value.label)




    # print( dataset[1].images)
    #
    # image_paths_flat = []
    # labels_flat = []
    # for i in range(len(dataset)):
    #     image_paths_flat += dataset[i].image_paths
    #     labels_flat += [i] * len(dataset[i].image_paths)

    # img = cv2.imread('D:\Downloads\omniglot-master\python\images_background_small1\Balinese\character01\\0108_01.png',0)
    # img_r = cv2.resize(img,(32,32),interpolation=cv2.INTER_LINEAR)
    # img_r = (img_r / 255.0 - 0.5) * 2
    # cv2.imshow('image', img)
    # cv2.imshow('img_r', img_r)
    # cv2.waitKey(0)
    # print(img_r.shape)
    # cv2.destroyAllWindows()




    # data = get_dataset('D:\Downloads\omniglot-master\python\images_background\Anglo-Saxon_Futhorc')
    # print(data[0].image_paths)





if __name__ == '__main__':
    main()
