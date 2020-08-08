import numpy as np
import matplotlib.image as im 

######################################################################
def conv(image, kernel, stride, padding, nlinear):
	#print image.shape
	if padding ==1:
		rimg = np.shape(image)[0]
		cimg = np.shape(image)[1]
		himg = np.shape(image)[2]
		image1 = np.zeros([rimg+((rimg-1)*stride - rimg + np.shape(kernel)[0]),cimg+((cimg-1)*stride - cimg + np.shape(kernel)[1]),himg])
		for i in range (2,rimg+((rimg-1)*stride - rimg + np.shape(kernel)[0])-2):
			for j in range(2,rimg+((rimg-1)*stride - rimg + np.shape(kernel)[1])-2):
				image1[i][j][:]=image[i-2][j-2][:]
		image = image1
	image1 = np.empty([np.shape(image)[0]-np.shape(kernel)[0]+1,np.shape(image)[1]-np.shape(kernel)[1]+1])
	for i in range(0,np.shape(image)[0]-np.shape(kernel)[0]):
		for j in range(0,np.shape(image)[1]-np.shape(kernel)[1]):
			total = 0
			for x in range(0,np.shape(kernel)[0]):
				for y in range(0,np.shape(kernel)[1]):
					total = total + np.sum(image[i+x][j+y]*kernel[x][y])
			image1[i][j]=total
			j = j+stride-1
		i = i+stride-1
	if nlinear == "sigmoid":
		result = sigmoid(image1)
	if nlinear == "relu":
		result = relu(image1)
	if nlinear == "tanh":
		result = tanh(image1)
	if nlinear == "none":
		result = image1
	#print image
	#print kernel
	#print result
	#print result.shape
	return result


def sigmoid(x):
	return 1/(1+np.exp(-x))

def relu(x):
	return x*(x>0)

def tanh(x):
	return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

def softmax(x):
	y = np.empty(x.shape)
	for i in range(0,np.shape(x)[0]):
		y[i] = (np.exp(x[i])/np.sum(np.exp(x)))
	return y

def softmaxprime(softmax):
    # Reshape the 1-d softmax to 2-d so that np.dot will do the matrix multiplication
    s = softmax.reshape(-1,1)
    return np.diagflat(s) - np.dot(s, s.T)

def reluprime(x):
	return 1*(x>0)
###########################################################
def pooling(conv,pool,stride): #pooling from 2x2 matrices
	if pool == "maximum":
		result=maxpool(conv,stride)
	if pool == "average":
		result = avgpool(conv,stride)
	return result

def maxpool(conv,stride):
	mx = np.empty([(conv.shape[0]-2)/stride+1,(conv.shape[1]-2)/stride+1])
	x=0
	y=0
	for i in range(0,(conv.shape[0]-2)/stride+1):
		for j in range(0,(conv.shape[1]-2)/stride+1):
			temp = np.array([[conv[i][j],conv[i][j+1]],[conv[i+1][j],conv[i+1][j+1]]])
			mx[x][y] = np.amax(temp)
			y=y+1
			j = j + stride - 1
		y = 0
		x+=1
		i = i + stride - 1
	return mx

def avgpool(conv,stride):
	avg = np.empty([np.shape(conv)[0]-(2-1)*stride,np.shape(conv)[1]-(2-1)*stride])
	x=0
	y=0
	for i in range(0,np.shape(conv)[0]-(2-1)*stride):
		for j in range(0,np.shape(conv)[1]-(2-1)*stride):
			temp = np.array([[conv[i][j],conv[i][j+1]],[conv[i+1][j],conv[i+1][j+1]]])
			avg[x][y] = np.sum(temp)/4
			y=y+1
			j = j + stride - 1
		y = 0
		x+=1
		i = i + stride - 1
	return avg

########################################################


def convlayer(image, kernel, stride, padding, nlinear):
	result = np.zeros([np.shape(kernel)[0],np.shape(image)[0],np.shape(image)[1]])
	for i in range (0,np.shape(kernel)[0]):
		result[i] = conv(image,kernel[i],stride, padding,nlinear)		
	size = np.shape(result)[0]*np.shape(result)[1]*np.shape(result)[2]
	#print "done"
	#print "Size = ", size

	#print image
	#print kernel
	#print result
	return result
########################################################

def poolvol(image,pool,stride):
	result = np.empty([np.shape(image)[0],image.shape[1]/2,image.shape[2]/2])
	for i in range(0,np.shape(image)[0]):
		result[i] = pooling(image[i],pool,stride)
	return result

#########################################################
def unraveling(acti_map,W1,b1):
	r = acti_map.shape[0]
	c = acti_map.shape[1]
	h = acti_map.shape[2]
	acti_map= acti_map.reshape(r*c*h,1)
	result = np.matmul(W1,acti_map)+b1
	return result


#########################################################

def mlp(vec,nh,sh,nlinear,so,W2,b2):
	for i in range(0,1):
		if nlinear == "sigmoid":
			vec = sigmoid(vec)
		if nlinear == "relu":
			vec = relu(vec)
		if nlinear == "tanh":
			vec = tanh(vec)
	out = np.matmul(vec.T,W2)+b2
	return out 




############################################################
def err(y,yhat):
	loss = 0
	y = y.reshape(10)
	yhat = yhat.reshape(10)
	for i in range(0,np.shape(y)[0]):
		loss = loss + y[i]*np.log(1/yhat)
	return loss
#########################################


def train(X_train,y_train,kernel1,kernel2,nlinear,pool,padding,stride,ab,cd):
	out = np.empty([np.shape(X_train)[0],1,10])
	for i in range(0,np.shape(X_train)[0]):	
		print i
		cl1 = convlayer(X_train[i], kernel1, 1, padding, nlinear)
		acti_map1 = poolvol(cl1,pool,2)
		acti_map1 = np.swapaxes(acti_map1,0,2)
		acti_map1 = np.swapaxes(acti_map1,0,1)
		cl2 = convlayer(acti_map1, kernel2, 1, padding, nlinear)
		acti_map2 = poolvol(cl2,pool,2)
		acti_map2 = np.swapaxes(acti_map2,0,2)
		acti_map2 = np.swapaxes(acti_map2,0,1)
		r,c,h = np.shape(acti_map2)
		if ab ==0:
			W1 = np.random.rand(1024,r*c*h)
			b1 = np.random.rand(1024,1)
			ab = 1
		vec = unraveling(acti_map2,W1,b1)
		vec = relu(vec)
		if cd ==0:
			W2 = np.random.rand(np.shape(vec)[0],10)
			b2 = np.random.rand(1,10)
			cd = 1
		out[i] = mlp(vec,1,np.shape(vec)[0],"relu",10,W2,b2)
		out[i] = out[i].reshape(10)
		out[i]=softmax(out[i])
		error = err(y_train[i],out[i])
		print out[i].shape
		if i%batch_size == 0:
			W1,W2,kernel1,kernel2 = update(out[i],y_train[i],vec,cl1,cl2,acti_map1,acti_map2,W1,W2,b1,b2,kernel1,kernel2)
	return out,ab,cd


################################################################################################

def update(yhat,y,vec,cl1,cl2,acti_map1,acti_map2,W1,W2,b1,b2,kernel1,kernel2):
	gama = 0.1
	O4 = acti_map2
	acti_map2 = acti_map2.reshape(acti_map2.shape[0]*acti_map2.shape[1]*acti_map2.shape[2],1)
	print W2.shape, W1.shape
	for i in range(0, W2.shape[0]):
		for j in range(0,W2.shape[1]):
			delta_loss = -(y[j]/yhat[0][j])*softmaxprime(yhat)[0][j]
			del_W2 = delta_loss*vec[i][0]
			W2[i][j] = W2[i][j] - gama*del_W2
			b2[0][j] = b2[0][j] - gama*delta_loss

	temp = np.empty([W1.shape[0],W1.shape[1]])
	for i in range(0,W1.shape[1]):
		for j in range(0,W1.shape[0]):
			delta_loss = np.sum(-(y/yhat[0])*softmaxprime(yhat)*W1[j][i])*reluprime(vec[j][0])
			del_W1 = delta_loss*acti_map2[i][0]
			temp[j][i] = del_W1
			W1[j][i] = W1[j][i] - gama*del_W1
			b1[j][0] = b1[j][0] - gama*delta_loss
	print temp.shape

	dv = np.matmul(temp.T,vec)
	dam2 = dv.reshape(7,7,64)            ###########################change change 
	dam2 = np.swapaxes(dam2,0,2)
	dam2 = np.swapaxes(dam2,1,2)	
	print dam2.shape
	dcl2 = np.zeros([cl2.shape[0],cl2.shape[1],cl2.shape[2]])
	#print dcl2.shape
	print cl2.shape

	x=0
	y=0

	print (cl2.shape[1]-2)/2+1, (cl2.shape[2]-2)/2+1
	for a in range(0,cl2.shape[0]):
		for i in range(0,(cl2.shape[1]-2)/2+1):
			for j in range(0,(cl2.shape[2]-2)/2+1):
				temp1 = np.array([[cl2[a][i][j],cl2[a][i][j+1]],[cl2[a][i+1][j],cl2[a][i+1][j+1]]])
				mx = np.amax(temp1)
				index = np.where(cl2 == mx)
				dcl2[index[0],index[1],index[2]] = dam2[a][x][y]
				print x,y,i,j
				y=y+1
				j = j + 2 - 1
			y = 0
			x+=1
			i = i + 2 - 1
		x=0
	#print dcl2.shape
	dcl2 = np.swapaxes(dcl2,0,2)
	dcl2 = np.swapaxes(dcl2,0,1)
	kernel22 = np.swapaxes(kernel2,0,3)
	dam1 = convlayer(dcl2, kernel22, 1, 1, "none")
	#print dam1.shape


	return W1, W2, kernel1, kernel2


###################################################################


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST data/", one_hot=True)
X_train = mnist.train.images
y_train = mnist.train.labels
X_test = mnist.test.images
y_test = mnist.test.labels
del mnist

X_trains = np.empty([(np.shape(X_train)[0]),28,28,1])
for i in range(0,np.shape(X_train)[0]):
	X_trains[i] = X_train[i].reshape(28,28,1)
X_train = X_trains
X_tests = np.empty([(np.shape(X_test)[0]),28,28,1])
for i in range(0,np.shape(X_test)[0]):
	X_tests[i] = X_test[i].reshape(28,28,1)
X_test = X_tests

l=2
stride = 1
padding = 1
nlinear = "relu"
pool = "maximum"
batch_size = 1


out = np.empty([np.shape(X_train)[0],1,10])
kernel1 = np.empty([32,5,5,1])
kernel2 = np.empty([64,5,5,32])
for i in range (0,np.shape(kernel1)[0]):
	kernel1[i] = np.random.rand(np.shape(kernel1)[1],np.shape(kernel1)[2],1)
for i in range (0,np.shape(kernel2)[0]):
	kernel2[i] = np.random.rand(np.shape(kernel2)[1],np.shape(kernel2)[2],np.shape(kernel2)[3])
ab = 0
cd = 0
losses = []
nepochs = 1

for i in range(0,nepochs):
	out,ab,cd = train(X_train,y_train,kernel1,kernel2,nlinear,pool,padding,stride,ab,cd)
	np.random.shuffle(X_train)  