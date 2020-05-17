# machine-learning-CNN

Machine learning exercise

A convolutional neural network that distinguishes even digits (0, 2, 4, 6, or 8) from the MNIST data set.

The main part of this project is 'main.py' . This script will read data from even_mnist.csv . This dataset is created from the MNIST dataset by fitlering out odd number. This leaves only 0, 2, 4, 6, and 8 to classify. The images are reduced from 28x28 to 14x14. Each row in the data file is a flattened 14x14 image and the correct label. The script also reads relevant parameters (e.g. learning rate, number of training epochs). The use needs to specify the path to the parameter file as a command line argument. To run this script use

```
python main.py param/param_file.json
```

The script can also be run in a 'help' mode. This mode displays the parameters specified in the parmater file and what each of these parameters are used for. To run the script in 'help' mode use

```
python main.py --help
```

This script reserves a specified number of rows to be used as test data and the rest is to be used as training data. The script trains a convolutional neural network that is given in 'nn_class.py' . The script loops over the specified number of training epochs and calculates the loss for both the training data and the test data. This loss is ouput into 'output.csv' .

The neural netork is specified in a class 'nn_class.py' . This neural network consists of two convolutional layers and two fully connected layers. The convolutional layers are pooled and then flattened before the data is passed to the fully connected layers. This neural network is trained using back propagation.
