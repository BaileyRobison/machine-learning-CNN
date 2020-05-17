# -*- coding: utf-8 -*-

import numpy as np
import sys, json
import torch
import torch.optim as optim
from nn_class import Net

torch.set_default_tensor_type('torch.DoubleTensor')

#get files paths from command line arguments
command_line_argument = sys.argv[1]

#print parameters from json file
if command_line_argument == '--help':
    print('Hyper parameters from json file')
    print('(parameter) \t (type) \t (default) \t (description)')
    print('learning rate \t float \t\t 0.001 \t\t learning rate of optimizer')
    print('momentum \t float \t\t 0.95 \t\t momentum of optimizer')
    print('num epoch \t int \t\t 200 \t\t number of training epochs')
    print('display epoch \t int \t\t 10 \t\t number of epochs to wait before console output')
    print('test size \t int \t\t 3000 \t\t size of test data set')
    sys.exit()
else: #argument is json file path
    json_file_path = command_line_argument

#read hyper parameters from json file
f = open(json_file_path, 'r')
hyper_params = json.load(f)
f.close()
alpha = hyper_params['learning rate']
momentum = hyper_params['momentum']
num_epoch = hyper_params['num epoch']
display_epoch = hyper_params['display epoch']
test_size = hyper_params['test size']


#read in data from csv file
csv_path = 'even_mnist.csv'
mnist_input = np.genfromtxt(csv_path, delimiter=' ')
xs = mnist_input[:,:-1] #extract 14x14 images
ys_column = mnist_input[:,-1] #extract digital values of images
ys = ys_column/2 #convert to indices

    
#training data set
x_train = torch.from_numpy( xs[test_size:].reshape(len(xs)-test_size,1,14,14) )
y_train = torch.from_numpy( ys[test_size:] )

#test data set
x_test = torch.from_numpy( xs[:test_size].reshape(test_size,1,14,14) )
y_test = torch.from_numpy( ys[:test_size] )


#create neural network
model = Net()

#define an optimizer and the loss function
optimizer = optim.SGD(model.parameters(), lr=alpha, momentum=momentum)
loss = torch.nn.NLLLoss(reduction = 'mean')
y_train = y_train.long()
y_test  = y_test.long()



train_vals=[]
test_vals=[]
for epoch in range(1, num_epoch + 1): #loop over epochs
    
    #train model and append to list
    train_val = model.backprop(x_train, y_train, loss, optimizer)
    train_vals.append(train_val)
    
    #test model and append to list
    test_val = model.test(x_test, y_test, loss)
    test_vals.append(test_val)
    
    #if this is a display epoch
    if epoch % display_epoch == 0:
        
        #calculate number that are correct
        with torch.no_grad():
            correct = 0 #initialize sum
            for i in range(len(x_test)): #loop over test data
                output = model.forward(x_test[i].reshape(1,1,14,14))
                target = y_test[i].reshape(-1)
                correct_index = torch.exp(output).max(1)[1] #index of max weight
                correct += (target==correct_index).item() #true if agrees with target
    
        #print loss and correct percentage
        print('epoch: '+str(epoch)+'/'+str(num_epoch)+\
                      '\tTraining Loss: '+'{:.4f}'.format(train_val)+\
                      '\tTest Loss: '+'{:.4f}'.format(test_val)+\
                      '\tCorrect '+str(correct)+'/'+str(len(x_test))+' ('+\
                      '{:.2f}'.format(100.*correct/len(x_test))+'%)')

#print final loss
print('Final training loss: '+'{:.4f}'.format(train_vals[-1]))
print('Final test loss: '+'{:.4f}'.format(test_vals[-1]))


#output
out_array = np.array([train_vals, test_vals]).T
with open('output.csv', 'wb') as f:
    f.write(b'train,test\n')
    np.savetxt(f, out_array, delimiter=',')





