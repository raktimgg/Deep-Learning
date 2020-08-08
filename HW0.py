import numpy as np
import matplotlib.image as im 

######################################################################
def conv(image, kernel, stride, padding, nlinear):
	print image.shape
	if padding ==1:
		rimg = np.shape(image)[0]
		cimg = np.shape(image)[1]
		himg = np.shape(image)[2]
		image1 = np.zeros([rimg+((rimg-1)*stride - rimg + np.shape(kernel)[0])/2-1,cimg+((cimg-1)*stride - cimg + np.shape(kernel)[1])/2-1,himg])
		print np.shape(image1)
		for i in range (1,((rimg-1)*stride - rimg + np.shape(kernel)[0])/2-np.shape(kernel)[0]):
			for j in range(1,((cimg-1)*stride - cimg + np.shape(kernel)[1])/2-np.shape(kernel)[1]):
				image1[i][j][:]=image[i-1][j-1][:]
		image = image1
	image1 = np.empty([np.shape(image)[0],np.shape(image)[1]])
	print np.shape(image)
	for i in range(0,np.shape(image)[0]-np.shape(kernel)[0]):
		for j in range(0,np.shape(image)[1]-np.shape(kernel)[1]):
			total = 0
			for x in range(0,np.shape(kernel)[0]):
				for y in range(0,np.shape(kernel)[1]):
					total = total + image[i+x][j+y][0]*kernel[x][y][0] + image[i+x][j+y][1]*kernel[x][y][1] + image[i+x][j+y][2]*kernel[x][y][2]
			image1[i][j]=total
			j = j+stride-1
		i = i+stride-1
	if nlinear == "sigmoid":
		result = sigmoid(image1)
	if nlinear == "relu":
		result = relu(image1)
	if nlinear == "tanh":
		result = tanh(image1)
	#print image
	#print kernel
	#print result
	print result.shape
	return result


def sigmoid(x):
	return 1/(1+np.exp(-x))

def relu(x):
	return x*(x>0)

def tanh(x):
	return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

###########################################################
def pooling(conv,pool): #pooling from 2x2 matrices
	if pool == "maximum":
		result=maxpool(conv)
	if pool == "average":
		result = avgpool(conv)
	return result

def maxpool(conv):
	mx = np.empty([np.shape(conv)[0]/2,np.shape(conv)[1]/2])
	x=0
	y=0
	for i in range(0,np.shape(conv)[0]/2):
		for j in range(0,np.shape(conv)[1]/2):
			temp = np.array([[conv[i][j],conv[i][j+1]],[conv[i+1][j],conv[i+1][j+1]]])
			mx[x][y] = np.amax(temp)
			y=y+1
		y = 0
		x+=1
	return mx

def avgpool(conv):
	avg = np.empty([np.shape(conv)[0]/2,np.shape(conv)[1]/2,np.shape(conv)[2]])
	x=0
	y=0
	for i in range(0,np.shape(conv)[0]/2):
		for j in range(0,np.shape(conv)[1]/2):
			temp = np.array([[conv[i][j],conv[i][j+1]],[conv[i+1][j],conv[i+1][j+1]]])
			avg[x][y] = np.sum(temp)/4
			y=y+1
		y = 0
		x+=1
	return avg

########################################################


def convlayer(image, kernel, stride, padding, nlinear):
	result = np.zeros([np.shape(kernel)[0],np.shape(image)[0],np.shape(image)[1]])
	for i in range (0,np.shape(kernel)[0]):
		result[i] = conv(image,kernel[i],stride, padding,nlinear)
	size = np.shape(result)[0]*np.shape(result)[1]*np.shape(result)[2]
	print "Size = ", size

	#print image
	#print kernel
	#print result
	return result
########################################################

def poolvol(image,pool):
	for i in range(np.shape(image)[0]):
		result = pooling(image[i],pool)
	return result

#########################################################
def unraveling(acti_map):
	r = acti_map.shape[0]
	c = acti_map.shape[1]
	acti_map= acti_map.reshape(np.shape(acti_map)[0]*np.shape(acti_map)[1],1)
	w = np.random.rand(r*c,1)
	result = acti_map*w
	return result


#########################################################

def mlp(vec,nh,sh,nlinear,so):
	for i in range(0,1):
		W = np.random.rand(sh,1)
		vec = vec*W
		print vec.shape
		if nlinear == "sigmoid":
			vec = sigmoid(vec)
		if nlinear == "relu":
			vec = relu(image1)
		if nlinear == "tanh":
			vec = tanh(image1)
	print vec.shape
	print W.shape
	W1 = np.random.rand(sh,so)
	print W1.shape
	out = np.matmul(vec.T,W1)
	return out 


print "Enter number of convolutional layers"
l = input()
#l = 2
print "Enter the image location inside double inverted commas "
loc = input() 
#loc = "download (1).jpeg"
image = im.imread(loc)
print "Enter number of strides"
stride = input()
#stride = 1
print "Enter 1 for padding, any other key for no-padding"
padding = input()
#padding = 0
print "Enter non-linearity inside double inverted commas. Choose among sigmoid, relu, tanh "
nlinear = input()
#nlinear = "sigmoid"
print "Enter pooling function inside double inverted commas. Choose among maximum and average"
pool = input()
#pool = "maximum"
kernel = np.empty([1,3,3,3])
for i in range (0,np.shape(kernel)[0]):
	kernel[i] = np.random.rand(np.shape(kernel)[1],np.shape(kernel)[2],3)
cl = convlayer(image, kernel, stride, padding, nlinear)
acti_map = poolvol(cl,pool)
vec = unraveling(acti_map)
out = mlp(vec,1,np.shape(vec)[0],"sigmoid",4)
print out.shape
print "output =", out 



############################################################

