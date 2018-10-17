import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (7,7)

def plot_learning_curves(x_axis_label, x_axis, f1_train, f1_test):
    plt.plot(x_axis, f1_train, marker='*')
    plt.plot(x_axis, f1_test, marker='*')
    plt.legend(['Train_F1_Scores', 'Test_F1_Scores'], loc='best')
    plt.xlabel(x_axis_label)
    plt.ylabel('F1 Score')
    plt.title('F1 Score v/s ' + x_axis_label)
    plt.show() 