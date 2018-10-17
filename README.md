Kindly download the MNIST dataset from http://yann.lecun.com/exdb/mnist/ and uncompress it; store it into a folder. In the usage its assumed to be in DATA_FOLDER. For detailed instructions on the project and for understanding the results, kindly read report.pdf. To read the problem statement, kindly refer to CS578_HW2.pdf


USAGE for vanilla_perceptron.py is

python vanilla_perceptron.py [size of training set] [no of epochs] [learning rate] [path to DATA_FOLDER]

for example
python vanilla_perceptron.py 10000 50 0.01 DATA_FOLDER

USAGE for average_perceptron.py is 

python average_perceptron.py [size of training set] [no of epochs] [learning rate] [DATA_FOLDER]

for example
python average_perceptron.py 10000 50 0.1 DATA_FOLDER

USAGE for winnow.py is

python winnow.py [size of training set] [no of epochs] [promotion/demotion factor] [threshold] [path to DATA_FOLDER]

for example
python winnow.py 10000 50 1.2 785 DATA_FOLDER
