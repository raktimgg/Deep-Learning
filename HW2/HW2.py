import numpy as np
import pandas as pd  #only used for importing csv values 
import matplotlib.pyplot as plt 

def onehot(x):
	out = np.zeros([x.shape[0],10])
	for i in range(0,x.shape[0]):
		temp = x[i]
		temp = int(temp)
		out[i][temp]=1.0
	return out


def loaddata():
	train = pd.read_csv('pendigits.tra', sep=',')
	train = np.array(train)
	test = pd.read_csv('pendigits.tes', sep=',')
	test = np.array(test)
	trainX = np.delete(train,16,1)
	testX = np.delete(test,16,1)
	trainY = np.empty([train.shape[0]])
	for i in range(trainY.shape[0]):
		trainY[i] = train[i][16]
	testY = np.empty([test.shape[0]])
	for i in range(testY.shape[0]):
		testY[i] = test[i][16]
	trainY = onehot(trainY)
	testY = onehot(testY)
	return trainX, trainY, testX, testY 

def sigmoid(x):
	x = np.array(x,dtype=np.float128)
	return 1/(1+np.exp(-x))

def relu(x):
	x = np.array(x,dtype=np.float128)
	for i in range (x.shape[1]):
		x[0][i] = np.max([0.1*x[0][i],x[0][i]])
	return x

def tanh(x):
	x = np.array(x,dtype=np.float128)
	return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

def softmax(x):
	x = np.array(x,dtype=np.float128)
	return np.exp(x) / np.sum(np.exp(x))

def softmaxprime(x):
    return softmax(x)*(1-softmax(x))

def reluprime(x):
	if x>0:
		x=1.0
	else:
		x=0.1
	return x

def batchnorm(X):
    mu = np.mean(X, axis=0)
    var = np.var(X, axis=0)

    X_norm = (X - mu) / np.sqrt(var + 1e-8)
    return(X_norm)
 
def sigmoidprime(x):
    return sigmoid(x)*(1-sigmoid(x))

def mlp(X,W1,b1,W2,b2):
	out1 = np.matmul(X,W1)+b1
	out1 = sigmoid(out1)
	#print out1.shape
	out2 = np.matmul(out1,W2)+b2
	out2 = softmax(out2)
	return out2,out1

def err(y,yhat):
	loss = 0
	y = y.reshape(10)
	yhat = yhat.reshape(10)
	for i in range(0,np.shape(y)[0]):
		if yhat[i]==0:
			loss = loss + y[i]*np.log(1/0.00001) 
		else:
			loss = loss + y[i]*np.log(1/yhat[i])
	return loss 

def update_sgd(acti_map2,vec,yhat,y,W1,b1,W2,b2):
	y = np.array(y,dtype=np.float128)
	yhat = np.array(yhat,dtype=np.float128)
	gama = 0.01
	for i in range(0, W2.shape[0]):
		for j in range(0,W2.shape[1]):
			#delta_loss = -(y[j]/yhat[0][j])*softmaxprime(yhat)[0][j]
			delta_loss = -(y[j]*(1-yhat[0][j]))
			del_W2 = delta_loss*vec[0][j]
			W2[i][j] = W2[i][j] - gama*del_W2
			b2[0][j] = b2[0][j] - gama*delta_loss

	temp = np.empty([W1.shape[0],W1.shape[1]])
	#print temp.shape, acti_map2.shape
	for i in range(0,W1.shape[1]):
		for j in range(0,W1.shape[0]):
			#delta_loss = np.sum(-(y/yhat[0])*softmaxprime(yhat)*W1[j][i])*sigmoidprime(vec[0][i])
			delta_loss = np.sum(-(y*(1-yhat)*W1[j][i]))*sigmoidprime(vec[0][i])	
			del_W1 = delta_loss*acti_map2[j]
			temp[j][i] = del_W1
			W1[j][i] = W1[j][i] - gama*del_W1
			b1[0][i] = b1[0][i] - gama*delta_loss
	#print W1
	return W1,b1,W2,b2	

def update_momentum(acti_map2,vec,yhat,y,W1,b1,W2,b2,dvW1,dvb1,dvW2,dvb2):
	gama = 0.01
	beta = 0.9 
	eta = 0.1
	for i in range(0, W2.shape[0]):
		for j in range(0,W2.shape[1]):
			#delta_loss = -(y[j]/yhat[0][j])*softmaxprime(yhat)[0][j]
			delta_loss = -(y[j]*(1-yhat[0][j]))
			del_W2 = delta_loss*vec[0][j]
			dvW2[i][j] = beta*dvW2[i][j] + eta*del_W2
			dvb2[0][j] = beta*dvb2[0][j] + eta*delta_loss
			W2[i][j] = W2[i][j] - gama*dvW2[i][j]
			b2[0][j] = b2[0][j] - gama*dvb2[0][j]

	temp = np.empty([W1.shape[0],W1.shape[1]])
	#print temp.shape, acti_map2.shape
	for i in range(0,W1.shape[1]):
		for j in range(0,W1.shape[0]):
			#delta_loss = np.sum(-(y/yhat[0])*softmaxprime(yhat)*W1[j][i])*sigmoidprime(vec[0][i])
			delta_loss = np.sum(-(y*(1-yhat)*W1[j][i]))*sigmoidprime(vec[0][i])	
			del_W1 = delta_loss*acti_map2[j]
			dvW1[j][i] = beta*dvW1[j][i] + eta*del_W1
			dvb1[0][i] = beta*dvb1[0][i] + eta*delta_loss
			W1[j][i] = W1[j][i] - gama*dvW1[j][i]
			b1[0][i] = b1[0][i] - gama*dvb1[0][i]
	return W1,b1,W2,b2,dvW1,dvb1,dvW2,dvb2	

def update_nest_momentum(acti_map2,vec,yhat,y,W1,b1,W2,b2,dvW1,dvb1,dvW2,dvb2):
	gama = 0.01
	beta = 0.9
	eta = 0.1
	for i in range(0, W2.shape[0]):
		for j in range(0,W2.shape[1]):
			#delta_loss = -(y[j]/yhat[0][j])*softmaxprime(yhat)[0][j]
			delta_loss = -(y[j]*(1-yhat[0][j]))
			del_W2 = delta_loss*vec[0][j]
			dvW2[i][j] = beta*dvW2[i][j] + eta*del_W2
			dvb2[0][j] = beta*dvb2[0][j] + eta*delta_loss
			W2[i][j] = W2[i][j] - gama*del_W2+beta*(beta*dvW2[i][j]-gama*delta_loss)
			b2[0][j] = b2[0][j] - gama*delta_loss+beta*(beta*dvb2[0][j]-gama*delta_loss)

	temp = np.empty([W1.shape[0],W1.shape[1]])
	#print temp.shape, acti_map2.shape
	for i in range(0,W1.shape[1]):
		for j in range(0,W1.shape[0]):
			#delta_loss = np.sum(-(y/yhat[0])*softmaxprime(yhat)*W1[j][i])*sigmoidprime(vec[0][i])
			delta_loss = np.sum(-(y*(1-yhat)*W1[j][i]))*sigmoidprime(vec[0][i])	
			del_W1 = delta_loss*acti_map2[j]
			dvW1[j][i] = beta*dvW1[j][i] + eta*del_W1
			dvb1[0][i] = beta*dvb1[0][i] + eta*delta_loss
			W1[j][i] = W1[j][i] - gama*del_W1+beta*(beta*dvW1[j][i]-gama*delta_loss)
			b1[0][i] = b1[0][i] - gama*delta_loss+beta*(beta*dvb1[0][i]-gama*delta_loss)
	return W1,b1,W2,b2,dvW1,dvb1,dvW2,dvb2

def update_adagrad(acti_map2,vec,yhat,y,W1,b1,W2,b2,r1,r2):
	epsilon = 0.01
	delta = 0.01
	for i in range(0, W2.shape[0]):
		for j in range(0,W2.shape[1]):
			#delta_loss = -(y[j]/yhat[0][j])*softmaxprime(yhat)[0][j]
			delta_loss = -(y[j]*(1-yhat[0][j]))
			del_W2 = delta_loss*vec[0][j]
			W2[i][j] = W2[i][j] - epsilon/(delta+np.sqrt(r2))*del_W2
			b2[0][j] = b2[0][j] - epsilon/(delta+np.sqrt(r2))*delta_loss
			r2 = r2 + delta_loss**2

	temp = np.empty([W1.shape[0],W1.shape[1]])
	#print temp.shape, acti_map2.shape
	for i in range(0,W1.shape[1]):
		for j in range(0,W1.shape[0]):
			#delta_loss = np.sum(-(y/yhat[0])*softmaxprime(yhat)*W1[j][i])*sigmoidprime(vec[0][i])
			delta_loss = np.sum(-(y*(1-yhat)*W1[j][i]))*sigmoidprime(vec[0][i])	
			del_W1 = delta_loss*acti_map2[j]
			temp[j][i] = del_W1
			W1[j][i] = W1[j][i] - epsilon/(delta+np.sqrt(r1))*del_W1
			b1[0][i] = b1[0][i] - epsilon/(delta+np.sqrt(r1))*delta_loss
			r1 = r1 + delta_loss**2
	return W1,b1,W2,b2,r1,r2	

def update_rmsprop(acti_map2,vec,yhat,y,W1,b1,W2,b2,r1,r2):
	epsilon = 0.01
	delta = 0.01
	p = 0.01
	for i in range(0, W2.shape[0]):
		for j in range(0,W2.shape[1]):
			#delta_loss = -(y[j]/yhat[0][j])*softmaxprime(yhat)[0][j]
			delta_loss = -(y[j]*(1-yhat[0][j]))
			del_W2 = delta_loss*vec[0][j]
			W2[i][j] = W2[i][j] - epsilon/(np.sqrt(delta+r2))*del_W2
			b2[0][j] = b2[0][j] - epsilon/(np.sqrt(delta+r2))*delta_loss
			r2 = (1-p)*r2 + p*delta_loss**2

	temp = np.empty([W1.shape[0],W1.shape[1]])
	#print temp.shape, acti_map2.shape
	#print W1.shape, b1.shape
	for i in range(0,W1.shape[1]):
		for j in range(0,W1.shape[0]):
			#delta_loss = np.sum(-(y/yhat[0])*softmaxprime(yhat)*W1[j][i])*sigmoidprime(vec[0][i])
			delta_loss = np.sum(-(y*(1-yhat)*W1[j][i]))*sigmoidprime(vec[0][i])	
			del_W1 = delta_loss*acti_map2[j]
			temp[j][i] = del_W1
			W1[j][i] = W1[j][i] - epsilon/(np.sqrt(delta+r1))*del_W1
			b1[0][i] = b1[0][i] - epsilon/(np.sqrt(delta+r1))*delta_loss
			r1 = (1-p)*r1 + p*delta_loss**2
	return W1,b1,W2,b2,r1,r2	

def update_adam(acti_map2,vec,yhat,y,W1,b1,W2,b2,r1,r2,s1,s2,t):
	epsilon = 0.01
	delta = 0.01
	p1 = 0.01
	p2 = 0.01
	for i in range(0, W2.shape[0]):
		for j in range(0,W2.shape[1]):
			#delta_loss = -(y[j]/yhat[0][j])*softmaxprime(yhat)[0][j]
			delta_loss = -(y[j]*(1-yhat[0][j]))
			del_W2 = delta_loss*vec[0][j]
			s2 = p1*s2 + (1-p1)*delta_loss
			r2 = (1-p2)*r2 + p2*delta_loss**2
			shat2 = s2/(1-p1**t)
			rhat2 = r2/(1-p2**t)
			W2[i][j] = W2[i][j] - epsilon/(np.sqrt(delta+rhat2))*shat2*del_W2
			b2[0][j] = b2[0][j] - epsilon/(np.sqrt(delta+rhat2))*shat2*delta_loss

	temp = np.empty([W1.shape[0],W1.shape[1]])
	#print temp.shape, acti_map2.shape
	for i in range(0,W1.shape[1]):
		for j in range(0,W1.shape[0]):
			#delta_loss = np.sum(-(y/yhat[0])*softmaxprime(yhat)*W1[j][i])*sigmoidprime(vec[0][i])
			delta_loss = np.sum(-(y*(1-yhat)*W1[j][i]))*sigmoidprime(vec[0][i])	
			del_W1 = delta_loss*acti_map2[j]
			temp[j][i] = del_W1
			s1 = p1*s1 + (1-p1)*delta_loss
			r1 = (1-p2)*r1 + p2*delta_loss**2
			shat1 = s1/(1-p1**t)
			rhat1 = r1/(1-p2**t)
			W1[j][i] = W1[j][i] - epsilon/(np.sqrt(delta+rhat1))*shat1*del_W1
			b1[0][i] = b1[0][i] - epsilon/(np.sqrt(delta+rhat1))*shat1*delta_loss
	return W1,b1,W2,b2,r1,r2,s1,s2

trainX, trainY, testX, testY =loaddata()
trainX = batchnorm(trainX)
testX = batchnorm(testX)
W1 = np.random.rand(trainX.shape[1], 10)
dvW1 = np.zeros([trainX.shape[1], 10])
b1 = np.random.rand(1 , 10)
dvb1 = np.zeros([1 , 10])
W2 = np.random.rand(10, 10)
dvW2 = np.zeros([10, 10])
b2 = np.random.rand(1 , 10)
dvb2 = np.zeros([1 , 10])
r1 = 0.01
r2 = 0.01
s1 = 0.01
s2 = 0.01
error = []
right = 0.0
accuracy = []

nEpochs = 20
for j in range (0,nEpochs):
	right = 0.0
	print j
	for i in range (0,trainX.shape[0]):
		out2, out1 = mlp(trainX[i],W1,b1,W2,b2)
		#print i
		#print err(trainY[i],out2)
		#print wr
		error.append(err(trainY[i],out2))
		if np.argmax(out2) == np.argmax(trainY[i]):
			right+=1
		W1,b1,W2,b2 = update_sgd(trainX[i],out1,out2,trainY[i],W1,b1,W2,b2)                                                   				#SGD 
		#W1,b1,W2,b2,dvW1,dvb1,dvW2,dvb2 = update_momentum(trainX[i],out1,out2,trainY[i],W1,b1,W2,b2,dvW1,dvb1,dvW2,dvb2)					#momentum
		#W1,b1,W2,b2,dvW1,dvb1,dvW2,dvb2 = update_nest_momentum(trainX[i],out1,out2,trainY[i],W1,b1,W2,b2,dvW1,dvb1,dvW2,dvb2)				#nesterov momentum
		#W1,b1,W2,b2,r1,r2 = update_adagrad(trainX[i],out1,out2,trainY[i],W1,b1,W2,b2,r1,r2)												#adagrad
		#W1,b1,W2,b2,r1,r2 = update_rmsprop(trainX[i],out1,out2,trainY[i],W1,b1,W2,b2,r1,r2)												#rmsprop
		#W1,b1,W2,b2,r1,r2, s1, s2 = update_adam(trainX[i],out1,out2,trainY[i],W1,b1,W2,b2,r1,r2,s1,s2,j+1)									#adam
	accuracy.append(right/trainX.shape[0])
x = np.linspace(0,nEpochs,nEpochs)
print accuracy
plt.plot(x,accuracy)
plt.show()


