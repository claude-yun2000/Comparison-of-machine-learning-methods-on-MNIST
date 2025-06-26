# Comparison-of-machine-learning-methods-on-MNIST
Q Yun (Dec 2024)

## Brief information
This project was to compare the performance of different machine learning methods, particularly between the traditional machine learning methods and deep learning, on the MNIST dataset. The code was run on Google Colab in Dec 2024. The techniques of cross validation and early stopping were sometimes implemented during the search for hyperparameters during the experiment.

The following introduction (together with the methodolgy) gives an overview of the project, and more information can be found in the Jupyter Notebook file in the repository.

### Introduction
The MNIST (Modified National Institute of Standards and Technology) dataset, with 70,000 grayscale images of handwritten 0-9 digits,  has been widely used for benchmarking different image recognition algorithms. Although convolutional neural network (CNN) and its variants are now widely considered the most effective models for image recognition, traditional machine learning algorithms tend to be more interpretable and may work well when the amount of data is limited. Therefore the advantages and disadvantages of both traditional machine learning algorithms and neural networks are worth exploring. This project will:
* Implement KNN (K-nearest neighbours), SVM (support vector machine), decision tree and random forest, ANN (artificial neural network) and CNN (including its variant of ResNet50) on MNIST dataset, and find the most appropriate model;
*	Investigate the impact of dimension reduction on the performance of  some models.

### Methodology
#### Data pre-processing
The dataset is split into a training set (the first 60,000 samples) and a test set (the last 10,000 samples) to ensure the comparability for other similar experiments. The training set may be further split into a train subset and a validation subset for some algorithms. A good practice is to check if all targeted classes are present in the training set and if the distribution of different classes are skewed (Gopalakrishnan, 2022). It can be seen though that the dataset are not skewed towards any particular digits.

It is often beneficial to scale the features(the 784 columns representing the pixels) to ensure that none of them have oversized impact on the models, and also to improve the convergence of gradient descend based algorithms. Two common feature scaling methods, standardisation (which centres the data at the mean of 0 and then rescale by the standard deviation of 1) and normalisation (which rescales the features within a range of 0 to 1) becomes the options. Since all pixel values fall within the range of 0 to 255 and the distributions of features are not necessarily Gaussian, the dataset will be normalised.

#### Models implemented
Although logistic regression can be extended to handle multiclass classification problems, it can be a bit clumsy to implement, while Linear discriminant analysis (LDA) assumes the data is normally distributed with the covariance matrix for all classes, which may not be the case. Therefore KNN, SVM, decision tree and random forest as the traditional machine learning algorithms will be implemented. For the deep learning models, a simple artificial neural network can serve as a benchmark to assess the performance of CNN (including ResNet50).

Dimension reduction may enhance both computational efficiency and model performance. Among all the reduction techniques, PCA (principal component analysis) is usually limited in handling non-linear data. T-sne (t-Distributed Stochastic Neighbour Embedding) clusters well on the MNIST dataset (see Naidu, 2021). However, after an initial effort to implement it, it was found that t-sne can’t be implemented consistently to the test data, and the transformation also proved to be computationally expensive, therefore it was excluded in this project. UMAP (Uniform Manifold Approximation and Projection) also handles non-linear data well, so will be implemented to assess its impact on some traditional machine learning models. Neural networks are already good at feature extraction, therefore they usually require no dimension reduction.

#### Metrics for model evaluation
While mean squared error is a good metric for regression tasks, accuracy, precision, recall and F1-score are more appropriate for classifications. Precision or recall may be more appropriate when the cost of false positive or false negative is high, but this is not a concern in identifying the digits in this project. Although accuracy can be misleading for imbalanced dataset, this is not the case for the MNIST dataset (as shown in the previous figure), therefore accuracy becomes a natural choice of the metric. Confusion matrices may help us identify the areas for improvement, hence will also be included to assist our analysis.

## Full report and the Python code (please refer to the Jupyter Notebook file)
