# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import h5py

print("Tensorflow Version:",tf.__version__)
print("Opencv Version:",cv2.__version__)
device_name = tf.test.gpu_device_name()
print(device_name)

#from google.colab import drive
#drive.mount('/content/drive')

def sigmoid(Z):
  A = 1/(1+np.exp(-Z))
  return A

def tanh(Z):
  A = (np.exp(Z)-np.exp(-Z))/(np.exp(Z)-np.exp(-Z))
  return A

def relu(Z):
  A = np.maximum(0,Z)
  return A

def load_data():
  test = h5py.File('drive/My Drive/Colab/Cat-vs-NonCat/test_catvnoncat.h5','r')
  train = h5py.File('drive/My Drive/Colab/Cat-vs-NonCat/train_catvnoncat.h5','r')
  #print(list(train.keys()))
  
  train_list_classes = list(train['list_classes'][:])
  train_set_x= np.array(train['train_set_x'][:])
  train_set_y= np.array(train['train_set_y'][:])
  
  test_list_classes = list(test['list_classes'][:])
  test_set_x= np.array(test['test_set_x'][:])
  test_set_y= np.array(test['test_set_y'][:])
  
  return train_set_x,train_set_y,test_set_x,test_set_y

train_set_x,train_set_y,test_set_x,test_set_y = load_data()
print("Train Shape:",train_set_x.shape,train_set_y.shape)
print("Test Shape:",test_set_x.shape,test_set_y.shape)

def reshape_data(train_set_x,train_set_y,test_set_x,test_set_y):
  train_set_x_reshaped = train_set_x.reshape(train_set_x.shape[0],-1).T
  train_set_y_reshaped = train_set_y.reshape(1,train_set_y.shape[0])
  
  test_set_x_reshaped = test_set_x.reshape(test_set_x.shape[0],-1).T
  test_set_y_reshaped = test_set_y.reshape(1,test_set_y.shape[0])
  
  train_set_x_flatten = train_set_x_reshaped/255
  test_set_x_flatten = test_set_x_reshaped/255
  
  return train_set_x_flatten,train_set_y_reshaped,test_set_x_flatten,test_set_y_reshaped

train_x,train_y,test_x,test_y = reshape_data(train_set_x,train_set_y,test_set_x,test_set_y)
print("Train Shape:",train_x.shape,train_y.shape)
print("Test Shape:",test_x.shape,test_y.shape)



np.random.seed(1)


def initialize_parameter(layer_dim):
  l = len(layer_dim)
  param = {}
  for i in range(1,l):
    param["W"+str(i)] = np.random.randn(layer_dim[i],layer_dim[i-1])/ np.sqrt(layer_dim[i-1])
    param["b"+str(i)] = np.zeros((layer_dim[i],1))
  return param

# init_param = initialize_parameter([2,4])
init_param = initialize_parameter([train_x.shape[0],20,7,5,1])
#print(init_param)
#print(len(init_param))

def linear_forward_activation(A_prev,W,b,activation):
  Z = np.dot(W,A_prev)+b
  if activation=="sigmoid":
    A = sigmoid(Z)
  elif activation=="relu":
    A = relu(Z)
  elif activation=="tanh":
    A = tanh(Z)
  cache = ((A_prev,W,b),Z)
  return A,cache

"""**RELU->RELU->RELU-->SIGMOID**"""

def linear_forward_n(train_x,param):
  num_layers = len(param)//2
  caches=[]
  #print("Number of layers:",num_layers)
  A = train_x
  for i in range(1,num_layers):
    A_prev = A
    A,cache = linear_forward_activation(A_prev,param["W"+str(i)],param["b"+str(i)],"relu")
    caches.append(cache)
    #print(A.shape)
  #print("\n",init_param["W"+str(num_layers)].shape,init_param["b"+str(num_layers)].shape)
  A_out,cache = linear_forward_activation(A,param["W"+str(num_layers)],param["b"+str(num_layers)],"sigmoid")
  caches.append(cache)
  return A_out,caches

A_out,caches = linear_forward_n(train_x,init_param)

print(len(caches))
#print(caches[3])

assert A_out.shape == train_y.shape

def compute_cost(A_out,train_y):
  m = A_out.shape[1]
  cost = np.sum((train_y*np.log(A_out))+((1-train_y)*np.log(1-A_out)))*(-1/m)
  return cost

compute_cost(A_out,train_y)

"""---**BACK PROP**---"""

def init_backward(train_y,A_out):
  #Last Layer dL/dA = dA_out
  dA_out = - (np.divide(train_y, A_out) - np.divide(1 - train_y, 1 - A_out))
  return dA_out

def sigmoid_backprop(dA,Z):
  A = sigmoid(Z)
  ## g'(z) = d/dz(g(z)) == a(1-a)
  dZ = dA*A*(1-A)
  return dZ
  

def relu_backprop(dA,Z):
  dZ = np.array(dA, copy=True) # just converting dz to a correct object.  
  # When z <= 0, you should set dz to 0 as well. 
  dZ[Z <= 0] = 0
  assert (dZ.shape == Z.shape)  
  return dZ

(A_prev,W,b),Z = cache[0]
print(A_prev)

def linear_backward(dA,cache,activation):
  
  (A_prev,W,b),Z = cache
  #print(A_prev)
  m = A_prev.shape[1]
  
  if activation == "sigmoid":
    dZ = sigmoid_backprop(dA,Z)
  elif activation == "relu":
    dZ =  relu_backprop(dA,Z)
  
  dW = 1./m * np.dot(dZ,A_prev.T)
  db = 1./m * np.sum(dZ, axis = 1, keepdims = True)
  dA_prev = np.dot(W.T,dZ)
  return dA_prev,dW,db

def backprop(train_y,A_out,caches):
  grads={}
  dA_out = init_backward(train_y,A_out)
  train_y = train_y.reshape(A_out.shape)
  L = len(caches)
  cache_last_layer = caches[L-1]
  lc,ac = cache_last_layer
  #print(lc,ac)
  #print(cache_last_layer)
  ##For Last Layer
  grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_backward(dA_out,cache_last_layer,"sigmoid")
  for l in reversed(range(L-1)):
    current_cache=caches[l]
    dA_curr = grads["dA" + str(l+1)]
    grads["dA" + str(l)], grads["dW" + str(l+1)], grads["db" + str(l+1)] = linear_backward(dA_curr,current_cache,"relu")
  return grads

grads = backprop(caches)

def update_parameters(parameters, grads, learning_rate):    
    L = len(parameters) // 2 # number of layers in the neural network

    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
        
    return parameters

init_param = update_parameters(init_param, grads, 0.0075)

def neural_net(train_x,train_y,layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
  np.random.seed(1)
  costs = []
  init_param = initialize_parameter(layers_dims)
  for i in range(0,num_iterations):
    A_out,caches = linear_forward_n(train_x,init_param)
    cost = compute_cost(A_out,train_y)
    grads = backprop(train_y,A_out,caches)
    init_param = update_parameters(init_param, grads, learning_rate)
    if print_cost and i%100==0:
      print ("Cost after iteration %i: %f" %(i, cost))
      costs.append(cost)
  return costs, init_param

layers_dims= [train_x.shape[0],20,7,5,1]
costs,param=neural_net(train_x,train_y,layers_dims, learning_rate = 0.0075, num_iterations = 2500, print_cost=True)

learning_rate = 0.0075
# plot the cost
plt.plot(np.squeeze(costs))
plt.ylabel('cost')
plt.xlabel('iterations (per tens)')
plt.title("Learning rate =" + str(learning_rate))
plt.show()

print(param)

def predict(X,Y,param):
  m = X.shape[1]
  n = len(param)//2
  p = np.zeros([1,m])
  A_out,caches = linear_forward_n(X,param)
  #print(A_out.shape)
  for i in range(A_out.shape[1]):
    if A_out[0,i]>0.5:
      p[0,i]=1
    else:
      p[0,i]=0
  accuracy = np.sum(p==Y)/m
  print("Accuracy",accuracy)
  #print(A_out)
  #print(p)
  #print(Y)

predict(test_x,test_y,param)

predict(train_x,train_y,param)

