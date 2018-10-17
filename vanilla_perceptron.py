import numpy as np
import sys
from util import *
# from graph_plotting import *

def train_vanilla(train_data, num_epoches, learning_rate):
    data_size = len(train_data)
    weights = np.random.uniform(-1,1,[10,785])
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
    
    return weights           

def main(argv):
    # learning_rate = 0.1
    # epochs = 50
    # train_data_size = 10000
    # path = '.'
    if len(argv) < 4:
        print "Usage: python vanilla_perceptron.py [size of training set] [no of epochs] [learning rate] [path to DATA_FOLDER]"
    else:
        train_data_size = int(argv[0])
        epochs = int(argv[1])
        learning_rate = float(argv[2])
        path = argv[3]

        train_data = preprocess(path + '/train-images.idx3-ubyte', path + '/train-labels.idx1-ubyte')
        test_data = preprocess(path + '/t10k-images.idx3-ubyte', path + '/t10k-labels.idx1-ubyte')

        vanilla_weights = train_vanilla(train_data[:train_data_size], epochs, learning_rate)
        f1_score_train = get_f1_score(train_data[:train_data_size], vanilla_weights)
        f1_score_test = get_f1_score(test_data, vanilla_weights)

        print "Training F1 Score: ", f1_score_train
        print "Test F1 Score: ", f1_score_test



    '''
    graph plotting
    '''
    # vanilla_epoch_f1_train = []
    # vanilla_epoch_f1_test = []
    # vanilla_epoch_x_axis = []
    # for i in range(10, 101, 5):
    #     vanilla_epoch_x_axis.append(i)
    #     vanilla_weights = train_vanilla(train_data[:10000], i, 0.001)
    #     f1_score_train = get_f1_score(train_data[:10000], vanilla_weights)
    #     f1_score_test = get_f1_score(test_data, vanilla_weights)
    #     vanilla_epoch_f1_train.append(f1_score_train)
    #     vanilla_epoch_f1_test.append(f1_score_test)
    # plot_learning_curves('Number of epochs', vanilla_epoch_x_axis, vanilla_epoch_f1_train, vanilla_epoch_f1_test)


if __name__ == '__main__':
    main(sys.argv[1:])