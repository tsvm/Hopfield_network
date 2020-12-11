#This is the sample code of discrere hopfield network


import numpy as np
import random
from PIL import Image
import os
import re

import matplotlib.pyplot as plt


#convert matrix to a vector
def mat2vec(x):
    m = x.shape[0]*x.shape[1]
    tmp1 = np.zeros(m)

    c = 0
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            tmp1[c] = x[i,j]
            c +=1
    return tmp1


#Create Weight matrix for a single image
def create_W(x):
    print("creating W...")
    if len(x.shape) != 1:
        print( "The input is not vector")
        return
    else:
        print('len(x)', len(x))
        w = np.zeros([len(x),len(x)])
        for i in range(len(x)):
            for j in range(i,len(x)):
                if i == j:
                    w[i,j] = 0
                else:
                    w[i,j] = x[i]*x[j]
                    w[j,i] = w[i,j]
    return w


#Read Image file and convert it to Numpy array
def readImg2array(file,size, threshold= 145):
    pilIN = Image.open(file).convert(mode="L")
    pilIN= pilIN.resize(size)
    #pilIN.thumbnail(size,Image.ANTIALIAS)
    imgArray = np.asarray(pilIN,dtype=np.uint8)
    x = np.zeros(imgArray.shape,dtype=np.float)
    x[imgArray > threshold] = 1
    x[x==0] = -1
    return x

#Convert Numpy array to Image file like Jpeg
def array2img(data, outFile = None):

    #data is 1 or -1 matrix
    y = np.zeros(data.shape,dtype=np.uint8)
    y[data==1] = 255
    y[data==-1] = 0
    img = Image.fromarray(y,mode="L")
    if outFile is not None:
        img.save(outFile)
    return img


#Update
def update(w,y_vec,theta=0.5,time=100,shape=None,counter=0):
    energies = []
    for s in range(time):
        m = len(y_vec)
        i = random.randint(0,m-1)
        u = np.dot(w[i][:],y_vec) - theta

        if u > 0:
            y_vec[i] = 1
        elif u < 0:
            y_vec[i] = -1

        if s % 2000 == 0:
            energy = -0.5 * y_vec @ w @ y_vec
            print("energy=", energy)
            energies.append(energy)

            outfile = f"{current_path}/after_{counter}_{s}.jpeg"
            temp_vec = y_vec.reshape(shape)
            temp_img = array2img(temp_vec)
            temp_img.show()

    return y_vec, energies


#The following is training pipeline
#Initial setting
def hopfield(train_files, test_files,theta=0.5, time=1000, size=(10,10),threshold=60, current_path=None):

    #read image and convert it to Numpy array
    print( "Importing images and creating weight matrix....")

    #num_files is the number of files
    num_files = 0
    for path in train_files:
        print( path)
        x = readImg2array(file=path,size=size,threshold=threshold)
        print('x:', x)
        x_vec = mat2vec(x)
        print('x_vec:', x_vec)
        print( len(x_vec))
        if num_files == 0:
            w = create_W(x_vec)
            print('w:', w)
            num_files = 1
        else:
            tmp_w = create_W(x_vec)
            print('tmp_w:', tmp_w)
            w = w + tmp_w
            print('w:', w)
            num_files +=1

    print( "Weight matrix is done!!")


    #Import test data
    counter = 0
    for path in test_files:
        y = readImg2array(file=path,size=size,threshold=threshold)
        oshape = y.shape

        outfile = current_path+"/before_"+str(counter)+".jpeg"
        y_img = array2img(y, outFile=outfile)
        y_img.show()
        print( "Imported test data")

        y_vec = mat2vec(y)
        print( "Updating...")
        print('y_vec', y_vec)
        y_vec_after, energies = update(w=w,y_vec=y_vec,theta=theta,time=time,shape=oshape,counter=counter)
        print('y_vec_after', y_vec_after)
        y_vec_after = y_vec_after.reshape(oshape)
        if current_path is not None:
            outfile = current_path+"/after_"+str(counter)+".jpeg"
            after_img = array2img(y_vec_after,outFile=outfile)
            after_img.show()
        else:
            after_img = array2img(y_vec_after,outFile=None)
            after_img.show()
        counter +=1

        # Plot energy change:
        plt.plot(energies)
        plt.ylabel('Energy')
        plt.show()


#Main
#First, you can create a list of input file path
current_path = os.getcwd()
train_paths = []
path = current_path+"/train_pics/"
for i in os.listdir(path):
    if re.match(r'[0-9a-zA-Z-]*.*g',i):
        train_paths.append(path+i)

#Second, you can create a list of sungallses file path
test_paths = []
path = current_path+"/test_pics/"
for i in os.listdir(path):
    if re.match(r'[0-9a-zA-Z-_]*.*g',i):
        test_paths.append(path+i)

#Hopfield network starts!
hopfield(train_files=train_paths, test_files=test_paths, theta=0,time=20000,size=(50,50),threshold=60, current_path = current_path)




