"""
Test the modules
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import torch
# import tensorflow as tf
import numpy as np
from glow import thops
from glow import modules
from glow import models


def is_equal(a, b, eps=1e-5):
    if a.shape != b.shape:
        return False
    max_delta = np.max(np.abs(a - b))
    return max_delta < eps


# def test_multidim_sum():
#     x = np.random.rand(2, 3, 4, 4)
#     th_x = torch.Tensor(x)
#     tf_x = tf.constant(x)
#     test_axis_list = [[1], [1, 2], [0, 2, 3], [0, 1, 2, 3]]
#     with tf.Session():
#         print("[Test] multidim sum, compared with tensorflow")
#         for axis in test_axis_list:
#             for keep in [False, True]:
#                 # tf
#                 tf_y = tf.reduce_sum(tf_x, axis=axis, keepdims=keep)
#                 tf_y = tf_y.eval()
#                 # th
#                 th_y = thops.sum(th_x, dim=axis, keepdim=keep).numpy()
#                 if is_equal(th_y, tf_y):
#                     print("  Pass: dim={}, keepdim={}", axis, keep)
#                 else:
#                     raise ValueError("sum with dim={} error".format(axis))


# def test_multidim_mean():
#     x = np.random.rand(2, 3, 4, 4)
#     th_x = torch.Tensor(x)
#     tf_x = tf.constant(x)
#     test_axis_list = [[1], [1, 2], [0, 2, 3], [0, 1, 2, 3]]
#     with tf.Session():
#         print("[Test] multidim mean, compared with tensorflow")
#         for axis in test_axis_list:
#             for keep in [False, True]:
#                 # tf
#                 tf_y = tf.reduce_mean(tf_x, axis=axis, keepdims=keep)
#                 tf_y = tf_y.eval()
#                 # th
#                 th_y = thops.mean(th_x, dim=axis, keepdim=keep).numpy()
#                 if is_equal(th_y, tf_y):
#                     print("  Pass: dim={}, keepdim={}", axis, keep)
#                 else:
#                     raise ValueError("mean with dim={} error".format(axis))


def test_actnorm():
    print("[Test]: actnorm")
    actnorm = modules.ActNorm2d(12)
    x = torch.Tensor(np.random.rand(2, 12, 64, 64))
    actnorm.initialize_parameters(x)
    y, det = actnorm(x, 0)
    x_, _ = actnorm(y, None, True)
    print("actnorm (forward,reverse) delta", float(torch.max(torch.abs(x_-x))))
    print("  det", float(det))


def test_conv1x1():
    print("[Test]: invconv1x1")
    conv = modules.InvertibleConv1x1(96)
    x = torch.Tensor(np.random.rand(2, 96, 16, 16))
    y, det = conv(x, 0)
    x_, _ = conv(y, None, True)
    print("conv1x1 (forward,reverse) delta", float(torch.max(torch.abs(x_-x))))
    print("  det", float(det))


def test_gaussian():
    # mean = torch.zeros((4, 32, 16, 16))
    # logs = torch.ones((4, 32, 16, 16))
    # x = torch.Tensor(np.random.rand(4, 32, 16, 16))
    # lh = modules.GaussianDiag.likelihood(mean, logs, x)
    # logp = modules.GaussianDiag.logp(mean, logs, x)
    pass


def test_flow_step():
    print("[Test]: flow step")
    step = models.FlowStep(32, 256, flow_coupling="affine")
    x = torch.Tensor(np.random.rand(2, 32, 16, 16))
    y, det = step(x, 0, False)
    x_, det0 = step(y, det, True)
    print("flowstep (forward,reverse)delta", float(torch.max(torch.abs(x_-x))))
    print("  det", det, det0)


def test_squeeze():
    print("[Test]: SqueezeLayer")
    layer = modules.SqueezeLayer(2)
    img = cv2.imread("pictures/tsuki.jpeg")
    img = cv2.resize(img, (256, 256))
    img = img.transpose((2, 0, 1))
    x = torch.Tensor([img])
    y, _ = layer(x, 0, False)
    x_, _ = layer(y, 0, True)
    z = y[0].numpy().transpose((1, 2, 0))
    cv2.imshow("0_3", z[:, :, 0: 3].astype(np.uint8))
    cv2.imshow("3_6", z[:, :, 3: 6].astype(np.uint8))
    cv2.imshow("6_9", z[:, :, 6: 9].astype(np.uint8))
    cv2.imshow("9_12", z[:, :, 9: 12].astype(np.uint8))
    cv2.imshow("x_", x_[0].numpy().transpose((1, 2, 0)).astype(np.uint8))
    cv2.imshow("x", x[0].numpy().transpose((1, 2, 0)).astype(np.uint8))
    cv2.waitKey()


def test_flow_net():
    print("[Test]: flow net")
    net = models.FlowNet((32, 32, 1), 128, 4, 3)
    x = torch.Tensor(np.random.rand(10, 1, 32, 32))

    pixels = thops.pixels(x)
    z = x + torch.normal(mean=torch.zeros_like(x),
                         std=torch.ones_like(x) * (1. / 256.))

    logdet = torch.zeros_like(x[:, 0, 0, 0])
    print('init logdet shape =', logdet.size())
    logdet += float(-np.log(256.) * pixels)
    print('init logdet value =', logdet[0])


    y, det = net(z,logdet)
    print("det : ", det)
    x_ = net(y, reverse=True)
    print("z", y.size())
    print("x_", x_.size())
    print("x_ :" , x_)
    print(torch.sum(torch.abs(x_ - z)))


nn_index = [-1,1]
def compute_index_logical_equal(inputs,indexs):
    logic_result = (inputs==indexs[0])
    for value in indexs:
        logic_result = np.logical_or(inputs == value,logic_result)
    return logic_result

def compute_index_logical_not_equal(inputs,indexs):
    logic_result = (inputs==indexs[0])
    for value in indexs:
        logic_result = np.logical_or(inputs == value,logic_result)
    return np.logical_not(logic_result)


import torch.nn as nn
from glow import thops
from glow import modules
from glow import models
from glow.config import JsonConfig


class AssociateGlowGenerated(nn.Module):
    def __init__(self, test_class_index, is_mean, K, class_number):
        super(AssociateGlowGenerated, self).__init__()
        self.glow = models.Glow(JsonConfig("hparams/omni_all_bg.json"), test_class_index=test_class_index , is_mean=is_mean, K=K, y_classes=class_number)
        self.class_number = class_number
        self.eval_index = test_class_index
        self.criterion = nn.MSELoss()
        self.ones_tensor = torch.ones((1, 32, 32)).float().cuda()

        self.glow_generate = models.Glow(JsonConfig("hparams/omni_all_bg.json"), test_class_index=test_class_index,
                                is_mean=is_mean, K=K, y_classes=class_number)

    def restore_glow_params(self,state_dict):
        self.glow.load_state_dict(state_dict)

    def resoter_glow_g_params(self,state_dict):
        self.glow_generate.load_state_dict(state_dict)


    def forward(self, inputs=None, labels=None, y_onehot=None, eps_std=None):

        x_shape = inputs.size()
        y_shape = labels.size()
        assert x_shape[0] == y_shape[0]
        #         print(labels)
        y_onehot = torch.FloatTensor(y_shape[0], self.class_number).cuda()  # if use gpu ,
        y_onehot.zero_()
        y_onehot.scatter_(1, labels.view(-1, 1), 1)
        filter_g_img = None
        #         print(y_onehot[0])


        #  generate image
        if self.training:
            with torch.no_grad():
                # decode input
                z, det = self.glow_generate(x=inputs, y_onehot=y_onehot, reverse=False)

                if not isinstance(labels, torch.Tensor):
                    labels = torch.from_numpy(labels)


                # z algorithm
                positive_index = compute_index_logical_equal(labels, self.eval_index)
                negitive_index = compute_index_logical_not_equal(labels, self.eval_index)
                #             print("positive_index:", positive_index)
                #             print("negitive_index:", negitive_index)

                z_negative = z[negitive_index]
                assert len(z_negative) % 2 == 0
                y_negative = labels[negitive_index]
                #             print('y_negative = ', y_negative)
                x_positive = z[positive_index]
                y_positive = labels[positive_index]
                #             print('y_positive = ', y_positive)

                sorted_y, indices_y = torch.sort(y_negative)
                z_negavite_sort = z_negative[indices_y]
                #             print('sorted_y = ', sorted_y)
                #             print('indices_y = ', indices_y)
                #             print('labels[indices_y] = ', labels[indices_y])
                #             print('y_negative[indices_y] = ', y_negative[indices_y])

                # sample
                z_size = z.size()
                batch_g_z_size = len(y_negative) // 2
                #             print('batch_g_size = ', batch_g_z_size)

                g_z = torch.zeros([batch_g_z_size, z_size[1], z_size[2], z_size[3]]).cuda()  # require_grad = False
                label_g_c = torch.zeros(batch_g_z_size).long().cuda()
                #             print('self.g_z = ', self.g_z.size())
                #             print('self.label_g_c = ', self.label_g_c.size())

                i = 0

                while i < batch_g_z_size:
                    for index_p in range(len(x_positive)):
                        g_z[i] = x_positive[index_p]

                        p = 0.6  # np.random.uniform(0.45, 0.5)
                        q = np.random.randint(0, 2)
                        scale = nn_index[q] * p

                        #                     start_index = np.random.randint(0,32)
                        #                     # start_index = 0
                        #                     end_index = start_index + 32
                        slic_index = np.random.permutation(64)[:20]
                        #                     print('scale = {},slic_index ={}'.format(scale,slic_index))
                        #                     print('y_positive[index_p] = {},y_negative[2 * i] ={}, y_negative[2 * i + 1] = {}'.
                        #                           format(y_positive[index_p],y_negative[indices_y][2 * i],y_negative[indices_y][2 * i + 1]))
                        #                     g_z[i][start_index:end_index] =  g_z[i][start_index:end_index] + \
                        #                                                                  scale * ( z_negavite_sort[2 * i][start_index:end_index] -
                        #                                                                                  z_negavite_sort[2 * i + 1][start_index:end_index])

                        g_z[i][slic_index] = g_z[i][slic_index] + scale * (
                                    z_negavite_sort[2 * i][slic_index] -
                                    z_negavite_sort[2 * i + 1][slic_index])

                        label_g_c[i] = y_positive[index_p]
                        i += 1
                        if i >= batch_g_z_size:
                            break

                #             print('label_g_c = ', label_g_c)
                # shuff_g_index = np.random.permutation(batch_g_z_size)
                # sg_z = .g_z[shuff_g_index]
                # .label_g_c = .label_g_c[shuff_g_index]

                # generate img

                y_onehot_g = torch.FloatTensor(batch_g_z_size, self.class_number).cuda()  # if use gpu ,
                y_onehot_g.zero_()
                y_onehot_g.scatter_(1, label_g_c.view(-1, 1), 1)

                g_img = self.glow_generate(z=g_z, y_onehot=y_onehot_g, eps_std=eps_std, reverse=True)
                # print('g_img requires_grad = ', g_img.requires_grad)
                g_img = torch.clamp(g_img, -1, 1)
                g_img = g_img.detach()
                #                 print('.g_img requires_grad = ', .g_img.requires_grad)

                filter_index_list = []
                # filter_g_label_list = []
                # filter_g_oneshot_list = []

                for i in range(len(g_img)):
                    g_corr_label = (labels == label_g_c[i])
                    #                     print('labels[g_corr_label ={}, label_g_c[i] = {}'.format(.label_g_c[i],labels[g_corr_label]))
                    diff_g = self.criterion(g_img[i], inputs[g_corr_label][0]) / self.criterion(
                        inputs[g_corr_label][0], self.ones_tensor)
                    #                     print('diff_g = ',diff_g)

                    if diff_g < 1:
                        filter_index_list.append(i)

                print('filter_index_ len  = ', len(filter_index_list))
                filter_g_img = g_img[filter_index_list]
                #                 print('filter_g_img shape = ', .filter_g_img.size())
                filter_g_label = label_g_c[filter_index_list]
                filter_g_oneshot = y_onehot_g[filter_index_list]

                # shuff_g_index = np.random.permutation(len(.filter_g_label))
                # .filter_g_img = .filter_g_img[shuff_g_index]
                # .filter_g_label = .filter_g_label[shuff_g_index]
                # .filter_g_oneshot = .filter_g_oneshot[shuff_g_index]

                if len(filter_g_img) != 0:
                    inputs = torch.cat((inputs, filter_g_img), 0)
                    print('input_real shape = ', inputs.size())
                    labels = torch.cat((labels, filter_g_label), 0)
                    print('label_real shape = ', labels.size())
                    y_onehot = torch.cat((y_onehot,filter_g_oneshot),0)
                    print('onehot_real shape = ', y_onehot.size())

                    shuff_g_index = np.random.permutation(len(inputs))
                    inputs = inputs[shuff_g_index]
                    labels = labels[shuff_g_index]
                    y_onehot= y_onehot[shuff_g_index]















        # classify
        z, det = self.glow(x=inputs, y_onehot=y_onehot, reverse=False)
        y_logits = self.glow.class_flow(z,y_onehot)
        return z, det, y_logits ,labels,filter_g_img



def test_glow():
    print("[Test]: Glow")
    from glow.config import JsonConfig
    glow = models.Glow(JsonConfig("hparams/celeba_test.json"),is_mean=True,test_class_index=[1,2],K=4,y_classes=10,arc_loss=True)
    # img = cv2.imread("pictures/tsuki.jpeg")
    # img = cv2.resize(img, (32, 32))
    # img = (img / 255.0).astype(np.float32)
    # img = img[:, :, ::-1].transpose(2, 0, 1)
    # x = torch.Tensor([img]*8)
    # glow.set_z_add_random()
    glow.cuda()

    x = torch.Tensor(np.random.rand(12, 1, 32, 32))
    print('x.size = ', x.size())

    batch_size = 12
    nb_digits = 10
    y = torch.LongTensor(batch_size).random_() % nb_digits
    print('y = ',y)
    print('y.view(-1,1) = ', y.view(-1,1))
    y_onehot = torch.FloatTensor(batch_size, nb_digits)
    y_onehot.zero_()
    y_onehot.scatter_(1, y.view(-1,1), 1)
    print('y_onehot:', y_onehot)


    z, det, y_logits = glow(x=x, y_onehot=y_onehot)
    y_logits = glow.class_flow(z,y_onehot)
    print('z.size() =  ',z.size())
    print('det = ',det)
    print('y_logits = ',y_logits)

    print(models.Glow.loss_generative(det))
    # print('y_logits =  ',y_logits)
    print(models.Glow.loss_class(y_logits,y))

if __name__ == "__main__":
    # test_multidim_sum()
    # test_multidim_mean()
    # test_actnorm()
    # test_conv1x1()
    # test_gaussian()
    # test_flow_step()
    # test_squeeze()
    # test_flow_net()
    test_glow()

    # a = torch.zeros([1,1,4])
    # b = torch.ones([1,1,1])
    # c = a+b
    # print(c)

    # weight = torch.Tensor([1, 2, 1, 1, 10])
    # loss_fn = torch.nn.CrossEntropyLoss()
    # input = torch.randn(3, 5)  # (batch_size, C)
    # target = torch.FloatTensor(3).random_(5).long()
    # loss = loss_fn(input, target)
    # print(input)
    # print(target)
    # print(loss)
