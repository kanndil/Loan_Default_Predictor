### Don't mind about this 
import warnings

warnings.filterwarnings('ignore')
###
import pandas as pd
import numpy as np


## Implementing NN Model with learning rate bias

import math
class Layer:
    
    ### activations
    def _identity(self,z):
        return z
    
    def _identity_diff(self,z):
        return np.ones_like(z)
    
    def _sigmoid(self,z):
        return (1/(1+np.exp(-1*z)))

    def _diff_sigmoid(self,z):
        return self._sigmoid(z)*(1-self._sigmoid(z))

    def _tanh(self,z):
        return (np.exp(z) - np.exp(-z))/(np.exp(z) + np.exp(-z))

    def _diff_tanh(self,z):
        return (1 - self._tanh(z) * self._tanh(z))
    
    ###########

    def __init__(self,n_input,n_output, activation="identity",name=None):
        self.n_output= n_output
        self.n_input= n_input
        self.name= name
        
        if activation == "identity":
            self.activation = self._identity
            self.diff_act= self._identity_diff
        
        elif activation == "sigmoid":
            self.activation = self._sigmoid
            self.diff_act= self._diff_sigmoid

        elif activation == "tanh":
            self.activation = self._tanh
            self.diff_act= self._diff_tanh
        
        
            
        
        
        self.W= np.random.randn(self.n_output,self.n_input)*np.sqrt(2/self.n_input)
        self.b= np.random.randn(self.n_output,1)*np.sqrt(2/self.n_input)

        self.dW= np.zeros_like(self.W)
        self.db= np.zeros_like(self.b)
        
        self.Z= None
        self.Ai = None
        
    
    def forward(self,Ai):
#         print("FWD")
#         print(Ai.shape)
#         print(self.W.shape)
#         print(self.b.shape)
        z =  np.add((self.W @ Ai),self.b)
        # print("1")
        # z = z.astype(float)
        # print("2")
#         print(z.shape)
        A = self.activation(z.astype(float))
#         print(A.shape)
        
        
        self.Z = z
        self.Ai = Ai
        return A
    
    
    def backward(self,inp):
        
#         print("input shape: ",end='')
#         print(inp.shape)
       
        act_diff = self.diff_act(self.Z)
#         print("act_diff shape: ",end='')
#         print(act_diff.shape)
        
        tmp = inp * act_diff
#         print("tmp shape: ",end='')
#         print(tmp.shape)
        
        bet = tmp @ self.Ai.T # vector of 1s
#         print("bet shape: ",end='')
#         print(bet.shape)
        
        
        e = np.ones((self.Ai.shape[1],1))
        db = tmp @ e
#         print("db shape: ",end='')
#         print(db.shape)
        
        self.dW = self.dW + bet
        self.db = self.db + db
        
        
        return self.W.T @ tmp
    
    
    def zeroing_delta(self):
        self.dW= np.zeros_like(self.W)
        self.db= np.zeros_like(self.b)
#########################################################################################################
class NN:
    
    ########
    ## losses
    def MSE(self,y,yhat):
        a=np.square(yhat-y)
        a=np.sum(a)
        b= 1/(2*y.shape[1])
        return a*b

    # diff losses
    def _diff_MSE(self,y,yhat):
        return (yhat-y)

    def _diff_MSE(self,y,yhat, factor):

        d=(yhat-y)
        return  pd.DataFrame(d.values*factor.values, columns=d.columns, index=d.index)
    
    #########
    
    def __init__(self,lr):
        self.layers = []
        self.alpha= lr
        
    
    
    
    def forward(self,inp):
        a=inp
#         print(a.shape)
        for layer in self.layers:
            a = layer.forward(a)
#             print(a.shape)
        
        return a
    
    def backward(self,input):
        gd = input
        for layer in self.layers[::-1]:
            gd = layer.backward(gd)
            
    def add_layer(self,n_input,n_output, activation="identity",name=None):
        self.layers.append(Layer(n_input,n_output, activation=activation,name=name))
        
    def fit_stochastic(self, x_train,y_train): #data dim is MxN .. M no of examples.. N no of dimension
        
        M = x_train.shape[0]
        
        
        x_train = x_train.T
        y_train = y_train.T
        
#         print(x_train.shape)
#         print(y_train.shape)
        
        for i in range(M):
            # if(i%10 == 0):
            #     print(i , " done")
            #print (i)
            #print("Epoche {}/{}".format(i+1,epochs))
            y_hat= self.forward(x_train)
            
            dl_dyhat = self._diff_MSE(y_train,y_hat)
            
            self.backward(dl_dyhat)
            # print(y_train.shape)
            # print("index i ",i," has an label of ",y_train[0,i])
           
            for i in range(len(self.layers)):
                # layers[i].dW=layers[i].dW/N
                # layers[i].db=layers[i].db/N
                if(y_train[0,i]):
                    learning_rate = self.alpha * 4
                else:
                    learning_rate = self.alpha

                self.layers[i].W = self.layers[i].W - learning_rate * (self.layers[i].dW/M)
                self.layers[i].b = self.layers[i].b - learning_rate * (self.layers[i].db/M)
                
            
            # zeroing deltas
            for layer in self.layers:
                layer.zeroing_delta()
            
        print("Finished....") 
            
    def fit(self, x_train,y_train, epochs=5): #data dim is MxN .. M no of examples.. N no of dimension
        

       M = x_train.shape[0]

       l = y_train.shape[0]
       biased_learning_rate = np.zeros(l)
       cc= y_train
       #print("y_train shape: ",y_train.shape)
       for i in range(l):
            #print("cc[",i,"] = ",cc[i])
            if (cc[i] == 0):
                biased_learning_rate[i]+= 1
            else:
                biased_learning_rate[i]+= 4
       biased_learning_rate= pd.DataFrame(biased_learning_rate)
       
        
       x_train = x_train.T
       y_train = y_train.T
        
       biased_learning_rate = biased_learning_rate.T
#         print(x_train.shape)
#         print(y_train.shape)
        
       for i in range(epochs):
           if(i%10 == 0):
            print("Epochs {}/{}".format(i+1,epochs))
           y_hat= self.forward(x_train)
        #    print("in _diff_MSE ")
           dl_dyhat = self._diff_MSE(y_train,y_hat,biased_learning_rate) 
           
        #    print("y_train shape ",y_train.shape)
        #    print("dl_dhat shape.iloc[:,0:111502] ",dl_dyhat.iloc[:,0:111502].shape)
        #    print("biased_learning_rate shape ",biased_learning_rate.shape)
        #    print("out _diff_MSE ")   
                 
           
           self.backward(dl_dyhat)
           
        
           # update using GD
           for i in range(len(self.layers)):
               # layers[i].dW=layers[i].dW/N
               # layers[i].db=layers[i].db/N
               self.layers[i].W = self.layers[i].W - self.alpha  * (self.layers[i].dW/(M) )
               self.layers[i].b = self.layers[i].b - self.alpha  * (self.layers[i].db/(M) )
               #print("(self.layers[i].db/M)",(self.layers[i].db/M))
                
            
           # zeroing deltas
           for layer in self.layers:
               layer.zeroing_delta()
            
       print("Finished....") 
    
    
    def predict(self,x_test): #data dim is NxD .. N no of examples.. D no of dimension
#         print(x_test.shape)
        y_hat= self.forward(x_test.T)
        #print(y_hat.shape)
        predicted_labels = y_hat.T
        print(predicted_labels.describe())
        y= float(predicted_labels.describe().iloc[4]) #25%
        df = [1 if x >= y  else 0 for x in predicted_labels[0]]
        return df
                    
        
                
        
        

###################
                    
nn = NN(lr=0.4)

nn.add_layer(25,8,activation="sigmoid",name="l1")
nn.add_layer(8,8,activation = "sigmoid",name="l2")
nn.add_layer(8,8,activation = "sigmoid",name="l3")
nn.add_layer(8,1,activation = "sigmoid",name="l4")


############
#nn.fit(X_train.astype(float),y_train.astype(float),epochs=200)

#predicted_labels = nn.predict(X_test)


#from sklearn.metrics import confusion_matrix
#print(confusion_matrix(y_test, predicted_labels))

#from sklearn.metrics import classification_report

#print(classification_report(y_test, predicted_labels))
#############