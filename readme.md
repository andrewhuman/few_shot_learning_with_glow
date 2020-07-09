Generalized category mixed distribution hypothesis
Complementary bias hypothesis for small sample

1 Introduction
In recent years, deep learning has achieved great success in image processing and natural language processing, such as VGG, resnet, etc.
However, most of the success is based on large-scale data, and learning from a few samples is still difficult.
On the contrary, humans can easily learn to identify new categories from several samples;
This shows that the existing algorithms are still far away from human intelligence.

It is generally believed that it is mainly because the deviation of the data is too large, when the data set is too small, which leads to overfitting.
But we think this is because the current network is not complete enough to extract the information in the data set.
We hope to be able to extract more information from the data, including previously considered invalid information, in other words, we think all information is useful.
We made two assumptions:
1 The mixed distribution of common attributes under the same general class tends to normal distribution
2 In the case of a small sample, different types of distribution deviations are also different

On the basis of the above two assumptions, by analyzing the data distribution of different sizes, we have drawn some interesting results.
On the basis of these, the deviation of data of different subcategories under the same major category can be supplemented. That is, under certain conditions, it is possible to transfer attribute changes of one class to another class, such as posture changes, integrity changes, and so on.
We proposed two hypotheses and made an experimental analysis. The results show that these two hypotheses can be used as a hypothesis basis for data expansion under a small sample.


2 Related Work

meta-learning or learning-to-learning:
In terms of small sample learning, meta-learning can use the learning experience on other tasks to quickly assist in learning new tasks. For example, learning to design neural structure, learning initialization parameters, learning design optimizer, learning design loss function, etc. Meta-learning classifier can be easily transferred to new learning tasks

metric learning
This method maps the samples to the metric space, so that the distance of similar samples in space is close, and the distance of heterogeneous samples is far away, so as to achieve better classification results. Including...

Generative and augmentation traning sample
Since the main problem is that there are too few samples, it is a natural idea to add training samples. This is mainly divided into two categories, one is to directly generate new samples, such as GAN-based, VAE-based, or directly synthesized, and the second is Feature enhancement, the main representatives of Attribute-guided augmentation (AGA), Multi-level Semantic Feature Augmentation, etc.
However, these algorithms rarely analyze the prior distribution of the data. 11 is to directly divide the training set in pairs according to the corresponding attributes, and then directly operate on the latent variables linearly. 12 is to train an attribute operation network

Whether these algorithms are linear or nonlinear, they operate on all attributes of all data sets without considering the prior distribution of the data
Operating on a latent variable whose distribution is unknown leads to the problem that it is impossible to distinguish which changes can be learned by positive samples, which changes cannot be learned by positive samples, and the range of changes cannot be determined.
We hope to make an estimation analysis of the prior distribution in the case of small samples to obtain the change and range of learnable attributes, and provide a basis for synthesizing new samples of high quality and different distributions.



# Glow
This is pytorch implementation of paper "Glow: Generative Flow with Invertible 1x1 Convolutions". Most modules are adapted from the offical TensorFlow version [openai/glow](https://github.com/openai/glow).



# Scripts
- Train a model with
    ```
    train.py <hparams> <dataset> <dataset_root>
    ```
- Generate `z_delta` and manipulate attributes with
    ```
    infer_celeba.py <hparams> <dataset_root> <z_dir>
    ```

# Training result
Currently, I trained model for 45,000 batches with `hparams/celeba.json` using CelebA dataset. In short, I trained with follwing parameters

|      HParam      |            Value            |
| ---------------- | --------------------------- |
| image_shape      | (64, 64, 3)                 |
| hidden_channels  | 512                         |
| K                | 32                          |
| L                | 3                           |
| flow_permutation | invertible 1x1 conv         |
| flow_coupling    | affine                      |
| batch_size       | 12 on each GPU, with 4 GPUs |
| learn_top        | false                       |
| y_condition      | false                       |



### Manipulate attribute
Use the method decribed in paper to calculate `z_pos` and `z_neg` for a given attribute.
And `z_delta = z_pos - z_neg` is the direction to manipulate the original image.





# Issues
There might be some errors in my codes. Please help me to figure out.
