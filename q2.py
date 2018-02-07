import numpy as np 

from sklearn.datasets import fetch_mldata
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

np.random.seed(1847)

class BatchSampler(object):
    '''
    A (very) simple wrapper to randomly sample batches without replacement.

    You shouldn't need to touch this.
    '''
    
    def __init__(self, data, targets, batch_size):
        self.num_points = data.shape[0]
        self.features = data.shape[1]
        self.batch_size = batch_size

        self.data = data
        self.targets = targets

        self.indices = np.arange(self.num_points)

    def random_batch_indices(self, m=None):
        '''
        Get random batch indices without replacement from the dataset.

        If m is given the batch will be of size m. Otherwise will default to the class initialized value.
        '''
        if m is None:
            indices = np.random.choice(self.indices, self.batch_size, replace=False)
        else:
            indices = np.random.choice(self.indices, m, replace=False)
        return indices 

    def get_batch(self, m=None):
        '''
        Get a random batch without replacement from the dataset.

        If m is given the batch will be of size m. Otherwise will default to the class initialized value.
        '''
        indices = self.random_batch_indices(m)
        X_batch = np.take(self.data, indices, 0)
        y_batch = self.targets[indices]
        return X_batch, y_batch  

class GDOptimizer(object):
    '''
    A gradient descent optimizer with momentum
    '''

    def __init__(self, lr, beta=0.0):
        self.lr = lr
        self.beta = beta
        self.delta_t = np.zeros(785)

    def sgd_momemtum(self):
        '''
        For use with question 2.1 only.
        :return: weights
        '''
        delta_t = [0]
        d_f = 0.02
        weight = [10.0]
        for i in range(199):
                # self.batch_sampler.get_batch() ## gets 1 from class init
                delta_t.append((-1) * self.lr * weight[i] * d_f + (self.beta * delta_t[i]))
                weight.append(weight[i] + delta_t[-1])


        return weight

    def update_params(self, weights, grad):
        # Update parameters using GD with momentum and return
        # the updated parameters
        # self.delta_t contains prev delta_t

        # update delta_t
        self.delta_t = (-1) * self.lr * grad + (self.beta * self.delta_t)

        # update weights
        updated_weights = weights + self.delta_t

        return updated_weights


class SVM(object):
    '''
    A Support Vector Machine
    '''

    def __init__(self, c, feature_count):
        self.c = c
        self.w = np.random.normal(0.0, 0.1, feature_count)
    def hinge_loss(self, X, y):
        '''
        Compute the hinge-loss for input data X (shape (n, m)) with target y (shape (n,)).

        Returns a length-n vector containing the hinge-loss per data point.
        '''
        # Implement hinge loss
        X = np.array(X)
        y = np.array(y)
        # print(self.w.T.shape)
        # print(X[0].shape)
        # print(X.shape)
        hl = []
        for i in range(len(X)):
            temp = 1 - y[i] * self.w.T.dot(X[i])
            hl.append(max(temp, 0))
            # hl.append(temp)

        return hl

    def grad(self, X, y):
        '''
        Compute the gradient of the SVM objective for input data X (shape (n, m))
        with target y (shape (n,))

        Returns the gradient with respect to the SVM parameters (shape (m,)).
        '''
        # Compute (sub-)gradient of SVM objective
        hl = self.hinge_loss(X, y)
        hl = np.array(hl) #100,1
        grad = []

        ## have N and C

        for i in range(len(X)):
            if hl[i] == 0:

                grad_sub2 = []
                for j in range(785):
                    if j == 0:
                        ## since column 1s bias was added, must not regularise bias
                        g = 0
                        grad_sub2.append(g)
                    else:
                        g = self.w[j]
                        grad_sub2.append(g)

                grad.append(grad_sub2)

                # grad.append(self.w)
            else:
                # since column 1s bias was added, must not regularise bias
                grad_sub = []
                for j in range(785):
                    if j == 0:
                        g = - y[i] * X[i][j]
                        grad_sub.append(g)
                    else:
                        g = self.w[j] - y[i] * X[i][j]
                        grad_sub.append(g)

                grad.append(grad_sub)

                # g = self.w - y[i] * X[i]
                # grad.append(g)

        grad = np.array(grad)
        # print('grad shapre')
        # print(grad.shape)
        return self.c * np.mean(grad, axis=0)

    def classify(self, X):
        '''
        Classify new input data matrix (shape (n,m)).

        Returns the predicted class labels (shape (n,))
        '''
        # Classify points as +1 or -1

        classify = []
        for i in range(len(X)):
            # print(np.sign(np.transpose(X[i]).dot(self.w)))
            classify.append(np.sign(np.transpose(X[i]).dot(self.w)))

        classify = np.array(classify)
        return classify

    def accuracy(self, y_true, y_pred):
        return accuracy_score(y_true, y_pred)


def load_data():
    '''
    Load MNIST data (4 and 9 only) and split into train and test
    '''
    mnist = fetch_mldata('MNIST original', data_home='./data')
    label_4 = (mnist.target == 4)
    label_9 = (mnist.target == 9)

    data_4, targets_4 = mnist.data[label_4], np.ones(np.sum(label_4))
    data_9, targets_9 = mnist.data[label_9], -np.ones(np.sum(label_9))

    data = np.concatenate([data_4, data_9], 0)
    data = data / 255.0
    targets = np.concatenate([targets_4, targets_9], 0)

    permuted = np.random.permutation(data.shape[0])
    train_size = int(np.floor(data.shape[0] * 0.8))

    train_data, train_targets = data[permuted[:train_size]], targets[permuted[:train_size]]
    test_data, test_targets = data[permuted[train_size:]], targets[permuted[train_size:]]
    print("Data Loaded")
    print("Train size: {}".format(train_size))
    print("Test size: {}".format(data.shape[0] - train_size))
    print("-------------------------------")
    return train_data, train_targets, test_data, test_targets

def optimize_svm(train_data, train_targets, penalty, optimizer, batchsize, iters):
    '''
    Optimize the SVM with the given hyperparameters. Return the trained SVM.
    '''

    bs = BatchSampler(train_data, train_targets, batchsize)
    # init weights randomly. Must update svm.w for GD
    svm = SVM(penalty, 785)

    for iter in range(iters):
        # print(iter)
        # get mini batch
        X_batch, y_batch = bs.get_batch(batchsize)

        # calculate gradient for minibatch
        grad = svm.grad(X_batch, y_batch)
        # print(grad)

        # update weights for this minibatch
        svm.w = optimizer.update_params(svm.w, grad)
        # print(svm.w)


    return svm

def SDG_momentum(lr, first_b, second_b):
    """
    Question 2.1 SGD With Momentum.
    Prints the plot wt for 200 time-steps using b= 0.0 and b= 0.9 on the same graph.

    :return: None
    """
    gdo_0 = GDOptimizer(lr, first_b)
    gdo_9 = GDOptimizer(lr, second_b)
    w_0 = gdo_0.sgd_momemtum()
    w_9 = gdo_9.sgd_momemtum()
    plt.plot(w_0, label='beta = 0.0')
    plt.plot(w_9, label='beta = 0.9')
    plt.xlabel("Time Steps")
    plt.ylabel("Weights parameter")
    plt.title('SGD with momemtum 0 vs 0.9')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    train_data, train_targets, test_data, test_targets = load_data()
    p = np.ones(11025)
    train_data_1 = np.vstack((p, train_data.T)).T
    q = np.ones(2757)
    test_data_1 = np.vstack((q, test_data.T)).T

    # 2.1
    SDG_momentum(1, 0.0, 0.9)

    ## 2.3.1 - 2.3.4
    ## b = 0.0
    optimizer = GDOptimizer(0.05, 0.0)
    svm_optimized = optimize_svm(train_data_1, train_targets, 1, optimizer, 100, 500)
    acc = svm_optimized.accuracy(train_targets, svm_optimized.classify(train_data_1))
    print('b = 0.0, Train accuracy:')
    print(acc)
    acc2 = svm_optimized.accuracy(test_targets, svm_optimized.classify(test_data_1))
    print("b = 0.0, Test accuracy:")
    print(acc2)

    ## hingeloss b = 0
    train_loss = np.mean(svm_optimized.hinge_loss(train_data_1, train_targets))
    print("b = 0. Train LOSS", train_loss)
    test_loss = np.mean(svm_optimized.hinge_loss(test_data_1, test_targets))
    print("b = 0. Test LOSS", test_loss)

    ## hingeloss b = 0.1
    optimizer2 = GDOptimizer(0.05, 0.1)
    svm_optimized2 = optimize_svm(train_data_1, train_targets, 1, optimizer2, 100, 500)
    acc3 = svm_optimized2.accuracy(train_targets, svm_optimized2.classify(train_data_1))
    print('b = 0.1, Train accuracy:')
    print(acc3)
    acc4 = svm_optimized2.accuracy(test_targets, svm_optimized2.classify(test_data_1))
    print("b = 0.1, Test accuracy:")
    print(acc4)

    # hingeloss b = 0.1
    train_loss2 = np.mean(svm_optimized2.hinge_loss(train_data_1, train_targets))
    print("b = 0.1. Train LOSS", train_loss2)
    test_loss2 = np.mean(svm_optimized2.hinge_loss(test_data_1, test_targets))
    print("b = 0.1. Test LOSS", test_loss2)


    ## 2.3.5
    ## Plot 28 x 28
    ## b = 0
    # pixels = svm_optimized.w[1:].reshape((28, 28))
    # plt.imshow(pixels, cmap='gray')

    ## b = 0.1
    pixels2 = svm_optimized2.w[1:].reshape((28, 28))
    plt.imshow(pixels2, cmap='gray')

    plt.show()