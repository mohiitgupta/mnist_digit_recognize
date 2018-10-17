import numpy as np
import sys
from util import *
# from graph_plotting import *

def train_winnow(train_data, num_epoches, factor, theta):
    data_size = len(train_data)
    # weights = np.ones((10,785))
    '''
    initialize weights for all 10 winnows randomly
    ''' 
    weights = np.random.uniform(0.1,1,[10,785])

    average_weights = np.zeros((10,785))
    bias = np.ones((data_size,1))
    for epoch in range(num_epoches):
        np.random.shuffle(train_data)
        examples, labels = get_features_labels(train_data, bias)
        for i,example in enumerate(examples):
            y_pred = np.sum(weights*example, axis = 1)
            for j in range(0,10):
                label = get_true_label(labels[i], j)
                if label == 1 and y_pred[j] < theta:
                    weights[j][example == 1] *= factor
                elif label == -1 and y_pred[j] >= theta:
                    weights[j][example == 1] /= factor
            average_weights += weights    
    return weights, average_weights           

def main(argv):
    if len(argv) < 5:
        print "Usage: python winnow.py [size of training set] [no of epochs] [promotion/demotion factor] [threshold] [path to DATA_FOLDER]"
    else:
        train_data_size = int(argv[0])
        epochs = int(argv[1])
        factor = float(argv[2])
        theta = float(argv[3])
        path = argv[4]
        

        train_data = preprocess(path + '/train-images.idx3-ubyte', path + '/train-labels.idx1-ubyte')
        test_data = preprocess(path + '/t10k-images.idx3-ubyte', path + '/t10k-labels.idx1-ubyte')
        # for i in range(10):
        # np.random.shuffle(train_data)
        weights, avg_weights = train_winnow(train_data[:train_data_size], epochs, factor, theta)
        f1_score_train = get_f1_score(train_data[:train_data_size], weights)
        f1_score_test = get_f1_score(test_data, weights)

        print "Training F1 Score: ", f1_score_train
        print "Test F1 Score: ", f1_score_test


    '''
    Plotting graphs
    '''
    # winnow_factor_f1_train = []
    # winnow_factor_f1_test = []
    # winnow_factor_x_axis = []
    # winnow_avg_factor_f1_train = []
    # winnow_avg_factor_f1_test = []
    # size = 10000 
    # print len(winnow_size_f1_test)
    # print len(winnow_factor_f1_train)
    # factor = 1.1
    # while factor <= 2.0:
    #     winnow_factor_x_axis.append(factor)
    #     winnow_weights, winnow_avg_weights = train_winnow(train_data[:size], 50, factor, 785)
    #     f1_score_train = get_f1_score(train_data[:size], winnow_weights)
    #     f1_score_test = get_f1_score(test_data, winnow_weights)
    #     print "f1 score test is ", f1_score_test
    #     winnow_factor_f1_train.append(f1_score_train)
    #     winnow_factor_f1_test.append(f1_score_test)
        
    #     winnow_avg_factor_f1_train.append(get_f1_score(train_data[:size], winnow_avg_weights))
    #     winnow_avg_factor_f1_test.append(get_f1_score(test_data, winnow_avg_weights))
    #     factor += 0.05
    # plot_learning_curves('Promotion/Demotion Factor', winnow_factor_x_axis, winnow_factor_f1_train, winnow_factor_f1_test)
    # plot_learning_curves('Promotion/Demotion Factor', winnow_factor_x_axis, winnow_avg_factor_f1_train, winnow_avg_factor_f1_test)


if __name__ == '__main__':
    main(sys.argv[1:])