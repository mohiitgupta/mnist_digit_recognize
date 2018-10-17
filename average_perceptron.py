import numpy as np
import sys
from util import *
# from graph_plotting import *

def train_average(train_data, num_epoches, learning_rate):
    data_size = len(train_data)
    '''
    initialize weights for all 10 perceptrons randomly
    ''' 
    weights = np.random.uniform(-1,1,[10,785])
    
    average_weights = np.zeros((10,785))
    bias = np.ones((data_size,1))
    for epoch in range(num_epoches):
        np.random.shuffle(train_data)
        examples, labels = get_features_labels(train_data, bias)
        for i,example in enumerate(examples):
            y_pred = np.sum(weights*example, axis = 1)
            for j in range(0,10):
                label = get_true_label(labels[i], j)
                if y_pred[j]*label < 0:
                    weights[j] += learning_rate*label*example
            average_weights += weights    
    return average_weights

def main(argv):
    # learning_rate = 0.1
    # epochs = 50
    # train_data_size = 10000
    # path = '.'
    if len(argv) < 4:
        print "Usage: python average_perceptron.py [size of training set] [no of epochs] [learning rate] [DATA_FOLDER]"
    else:
        train_data_size = int(argv[0])
        epochs = int(argv[1])
        learning_rate = float(argv[2])
        path = argv[3]

        train_data = preprocess(path + '/train-images.idx3-ubyte', path + '/train-labels.idx1-ubyte')
        test_data = preprocess(path + '/t10k-images.idx3-ubyte', path + '/t10k-labels.idx1-ubyte')

        avg_weights = train_average(train_data[:train_data_size], epochs, learning_rate)
        f1_score_train = get_f1_score(train_data[:train_data_size], avg_weights)
        f1_score_test = get_f1_score(test_data, avg_weights)

        print "Training F1 Score: ", f1_score_train
        print "Test F1 Score: ", f1_score_test

    '''
    graph plotting
    '''
    # average_lr_f1_train = []
    # average_lr_f1_test = []
    # average_lr_x_axis = []
    # i=0.0001
    # while i < 1:
    #     average_lr_x_axis.append(i)
    #     average_weights = train_average(train_data[:10000], 50, i)
    #     f1_score_train = get_f1_score(train_data[:10000], average_weights)
    #     f1_score_test = get_f1_score(test_data, average_weights)
    #     average_lr_f1_train.append(f1_score_train)
    #     average_lr_f1_test.append(f1_score_test)
    #     i*=10
    # plot_learning_curves('Learning rate', average_lr_x_axis, average_lr_f1_train, average_lr_f1_test)


if __name__ == '__main__':
    main(sys.argv[1:])