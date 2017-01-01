# Udacity-Deeplearning"
These are my homeworks and solutions for the Udacity deeplearning course.
## Software requirements
- Anaconda Python 2.7 or Python 2.7 with at least numpy, scikit learn, pandas, matplotlib - I use the Anaconda stack therefore I not proved it in detail
- Google Tensorflow r.011 or newer
- best a GPU like GTX 1060, GTX 1070 and so on


## Notebooks
- 1_notmnist.ipynb: The objective of this assignment is to learn about simple data curation practices, and familiarize you with some of the data we'll be reusing later. This notebook uses the notMNIST dataset to be used with python experiments. This dataset is designed to look like the classic MNIST dataset, while looking a little more like real data: it's a harder task, and the data is a lot less 'clean' than MNIST. Mostly loading and preprocessing data. Then a simple logistic regression classifier.,
- 2_fullyconnected.ipynb: The goal of this assignment is to progressively train deeper and more accurate models using TensorFlow. First going to train a multinomial logistic regression using simple gradient descent.Then turn the logistic regression example with SGD into a 1-hidden layer neural network with rectified linear units nn.relu() and 1024 hidden nodes. 
- 3_regularization.ipynb: The goal of this assignment is to explore regularization techniques.Introduce and tune L2 regularization for both logistic and neural network models. Then introduce Dropout on the hidden layer of the neural network. Remember: Dropout should only be introduced during training, not evaluation, otherwise your evaluation results would be stochastic as well. TensorFlow provides nn.dropout() for that, but you have to make sure it's only inserted during training. Finally try to get the best performance you can using a multi-layer model! I achieved 95.5%.
- 4_convolutions.ipynb: This notebook explores CNNs (Convolution Neural Networks) to the notMNIST dataset using Tensorflow. First  build a small network with two convolutional layers, followed by one fully connected layer. Explore max pooling. Then build a CNN with more layers to get best test accuracy. I achieved test accuracy: 95.7% 
- 5_word2vec.ipynb: This notebook is about text anaylsis tools. The goal of this assignment is to train a Word2Vec skip-gram model over Text8 data and then implement and test an alternative to skip-gram: a model called CBOW (Continuous Bag of Words). In the CBOW model, instead of predicting a context word from a word vector, you predict a word from the sum of all the word vectors in its context. Implement and evaluate a CBOW model trained on the text8 dataset.




