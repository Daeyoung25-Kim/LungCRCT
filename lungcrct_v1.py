# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score,recall_score, f1_score, precision_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization,InputLayer,Conv2DTranspose,UpSampling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.initializers import HeNormal, GlorotNormal

tf.__version__ #implemented 2.19.0

import kagglehub

kagglehub.__version__ #0.3.13

os.environ['KAGGLEHUB_CACHE'] = '/tmp/kagglehub_cache'

import kagglehub

# Download latest version
path = kagglehub.dataset_download("hamdallak/the-iqothnccd-lung-cancer-dataset")

print("Path to dataset files:", path)

tr_dir_N = '/tmp/kagglehub_cache/datasets/hamdallak/the-iqothnccd-lung-cancer-dataset/versions/1/The IQ-OTHNCCD lung cancer dataset/Normal cases'
tr_dir_B = '/tmp/kagglehub_cache/datasets/hamdallak/the-iqothnccd-lung-cancer-dataset/versions/1/The IQ-OTHNCCD lung cancer dataset/Bengin cases'
tr_dir_M = '/tmp/kagglehub_cache/datasets/hamdallak/the-iqothnccd-lung-cancer-dataset/versions/1/The IQ-OTHNCCD lung cancer dataset/Malignant cases'

#If already downloaded, set directory as below

#tr_dir_N = '/kaggle/input/the-iqothnccd-lung-cancer-dataset/The IQ-OTHNCCD lung cancer dataset/Normal cases'
#tr_dir_B = '/kaggle/input/the-iqothnccd-lung-cancer-dataset/The IQ-OTHNCCD lung cancer dataset/Bengin cases'
#tr_dir_M = '/kaggle/input/the-iqothnccd-lung-cancer-dataset/The IQ-OTHNCCD lung cancer dataset/Malignant cases'

Filename1 = pd.DataFrame(os.listdir(tr_dir_N),columns=["filename"])
Filename2 = pd.DataFrame(os.listdir(tr_dir_B),columns=["filename"])
Filename3 = pd.DataFrame(os.listdir(tr_dir_M),columns=["filename"])

Filename1 = Filename1.sort_values(by=['filename'], ascending=True)
Filename2 = Filename2.sort_values(by=['filename'], ascending=True)
Filename3 = Filename3.sort_values(by=['filename'], ascending=True)

Filename1.index = [i for i in range(0,416,1)]
Filename2.index = [i for i in range(0,120,1)]
Filename3.index = [i for i in range(0,561,1)]

def load_images_and_labels(image,label):
    IMG = []
    LB = []
    for index, row in label.iterrows():
        img_path = os.path.join(image, f"{row['filename']}")
        if os.path.exists(img_path):
            img = cv2.imread(img_path,cv2.IMREAD_COLOR_RGB)
            img = cv2.resize(img, (256, 256)) # Resize images 256 x 256
            IMG.append(img)
            if "Normal" in row['filename']:
              LB.append(0)
            elif "Bengin" in row['filename']:
              LB.append(1)
            else:
              LB.append(2)
            if len(LB)%100 == 0: print(index+1)
    return np.array(IMG), np.array(LB)

# Loading images and labels
X_n,tag_n = load_images_and_labels(tr_dir_N,Filename1)
X_b,tag_b = load_images_and_labels(tr_dir_B,Filename2)
X_m,tag_m = load_images_and_labels(tr_dir_M,Filename3)

#############Train data

#random index for train data sampling

np.random.seed(4321)
idx1 = np.random.choice([i for i in range(0,416,1)],80,replace=False)
idx2 = np.random.choice([i for i in range(0,120,1)],80,replace=False)
idx3 = np.random.choice([i for i in range(0,561,1)],80,replace=False)

idx1

X_tr_n = X_n[idx1,:,:,:]
tag_tr_n = tag_n[idx1]

X_tr_b = X_b[idx2,:,:,:]
tag_tr_b = tag_b[idx2]

X_tr_m = X_m[idx3,:,:,:]
tag_tr_m = tag_m[idx3]

X_tr1 = np.concatenate([X_tr_n,X_tr_b,X_tr_m],axis=0)
tag_tr1 = np.concatenate([tag_tr_n,tag_tr_b,tag_tr_m],axis=0)

###Shuffle train data

X_tr1 = X_tr1/255.0 #rescaling

#shuffling data
(X_2,X_3,tag_2,tag_3) = train_test_split(X_tr1,tag_tr1,test_size=0.5,shuffle=True,random_state=321)
X_tr = np.concatenate([X_2,X_3],axis=0)
tag_tr = np.concatenate([tag_2,tag_3],axis=0)

#Example of train dataset
plt.imshow(X_tr[4,:,:,:])
print(tag_tr[4])

########### Test data sampling

###Test data



X_te_n = np.delete(X_n,idx1,axis=0)
tag_te_n = np.delete(tag_n,idx1)

X_te_b = np.delete(X_b,idx2,axis=0)
tag_te_b = np.delete(tag_b,idx2)

X_te_m = np.delete(X_m,idx3,axis=0)
tag_te_m = np.delete(tag_m,idx3)

np.random.seed(321321)
inx_1 = np.random.choice([i for i in range(0,(416-80),1)],40,replace=False) #40
X_te_n1 = X_te_n[inx_1,:,:,:]
tag_te_n1 = tag_te_n[inx_1]

np.random.seed(321321)
inx_3 = np.random.choice([i for i in range(0,(561-80),1)],40,replace=False) #40
X_te_m1 = X_te_m[inx_3,:,:,:]
tag_te_m1 = tag_te_m[inx_3]

X_te1 = np.concatenate([X_te_n1,X_te_b,X_te_m1],axis=0)
tag_te1 = np.concatenate([tag_te_n1,tag_te_b,tag_te_m1],axis=0)

###Shuffle Training data
X_te1 = X_te1/255.0

#shuffling data
(X__,X__2,tag__,tag__2) = train_test_split(X_te1,tag_te1,test_size=0.4,shuffle=True,random_state=321)
X_te = np.concatenate([X__,X__2],axis=0)
tag_te = np.concatenate([tag__,tag__2],axis=0)

np.bincount(tag_te)

#Example of cropped dataset
plt.imshow(X_tr[32,38:218,38:218,:])
print(tag_tr[32])

######Cropping dataset to 180 x 180

X_tr2 = X_tr[:,38:218,38:218,:]
X_te2 = X_te[:,38:218,38:218,:]

k_=7
plt.imshow(X_te2[k_,:,:,:])
print(tag_te[k_])

#################Defining LungCRCT_framework



#LungCRCT framework

tf.keras.utils.set_random_seed(321)

class lungcrct(tf.keras.Model):
  def __init__(self,input_dim, latent_dim1,latent_dim2,h1,h2,h3,h4,d1,d1_2,d2):
    super(lungcrct, self).__init__()
    self.input_dim = input_dim
    self.latent_dim1 = latent_dim1
    self.latent_dim2 = latent_dim2
    self.h1 = h1
    self.h2 = h2
    self.h3 = h3
    self.h4 = h4
    self.d1 = d1
    self.d1_2 = d1_2
    self.d2 = d2


    k = tf.keras.initializers.HeNormal(123)
    self.Enc = Sequential([
        InputLayer(shape=(input_dim,input_dim,3)),
        Conv2D(filters=h1,kernel_size=6,strides=(2,2),activation="silu"),
        Conv2D(filters=h2,kernel_size=6,strides=(2,2),activation="silu"),
        Conv2D(filters=h3,kernel_size=4,strides=(2,2),activation="silu"),
        Conv2D(filters=h4,kernel_size=4,strides=(2,2),activation="silu"),
        Conv2D(filters=h4,kernel_size=3,strides=(2,2),activation="silu"),
        Flatten(),
        Dense(units = d1,activation="elu",kernel_initializer=k),
        Dense(units = d1_2,activation="elu",kernel_initializer=k),
        Dense(units=(latent_dim1+latent_dim2)*2,kernel_initializer=k,activation="linear")
        ])

    self.Dec = Sequential([
        InputLayer(shape=(latent_dim1+latent_dim2,)),
        Dense(units=d1_2, activation="elu",kernel_initializer=k),
        Dense(units = d1,activation="elu",kernel_initializer=k),
        Dense(units=4*4*h4, activation="elu",kernel_initializer=k),
        tf.keras.layers.Reshape(target_shape=(4,4,h4)),
        Conv2DTranspose(filters=h4, kernel_size=3, strides=(2,2), activation='silu',kernel_initializer=k),
        Conv2DTranspose(filters=h3, kernel_size=4, strides=(2,2), activation='silu',kernel_initializer=k),
        Conv2DTranspose(filters=h2, kernel_size=4, strides=(2,2), activation='silu',kernel_initializer=k),
        Conv2DTranspose(filters=h1*2, kernel_size=6, strides=(2,2), activation='silu',kernel_initializer=k), #*2
        Conv2DTranspose(filters=3, kernel_size=6, strides=(2,2), kernel_initializer=k,activation='sigmoid'),
        ])



    mat1 = np.zeros((latent_dim2+1)*(latent_dim2+1)*d2).reshape(latent_dim2+1,(latent_dim2+1)*d2)
    mat2 = np.zeros((latent_dim2+1)*d2*(latent_dim2+1)*d2).reshape((latent_dim2+1)*d2,(latent_dim2+1)*d2)
    mat3 = np.zeros((latent_dim2+1)*d2*(latent_dim2+1)).reshape((latent_dim2+1)*d2,(latent_dim2+1))
    mat0 = np.ones((latent_dim2+1,latent_dim2+1)).reshape(latent_dim2+1,-1)

    for i in range(latent_dim2+1):
      mat1[i,(d2*i):(d2*(i+1))] = 1
      mat2[(i*d2):((i+1)*d2),(d2*i):(d2*(i+1))] = 1
      mat3[(i*d2):((i+1)*d2),i] = 1
      mat0[i,i] = 0

    mat0[-1,:] = 0

    mask1 = np.array(mat1).reshape(latent_dim2+1,(latent_dim2+1)*d2)
    class mask_1(tf.keras.constraints.Constraint):
      def __call__(self,w):
        return tf.convert_to_tensor(mask1)*w


    mask2 = np.array(mat2).reshape((latent_dim2+1)*d2,(latent_dim2+1)*d2)
    class mask_2(tf.keras.constraints.Constraint):
      def __call__(self,w):
        return tf.convert_to_tensor(mask2)*w

    mask3 = np.array(mat3).reshape((latent_dim2+1)*d2,latent_dim2+1)
    class mask_3(tf.keras.constraints.Constraint):
      def __call__(self,w):
        return tf.convert_to_tensor(mask3)*w

    mask0 = np.array(mat0).reshape(latent_dim2+1,latent_dim2+1)
    class mask_0(tf.keras.constraints.Constraint):
      def __call__(self,w):
        return tf.convert_to_tensor(mask0)*w


    k2 = tf.keras.initializers.HeNormal(123)
    self.G_ENC = Sequential([
        InputLayer(shape=(latent_dim2+1,)),
        Dense(units = (latent_dim2+1)*d2, use_bias=False, kernel_initializer=k2,activation="elu",kernel_constraint=mask_1()),
        Dense(units = (latent_dim2+1)*d2, use_bias=False,kernel_initializer=k2,activation="elu",kernel_constraint=mask_2()),
        Dense(units=latent_dim2+1,use_bias=False,kernel_initializer=k2,activation="linear",kernel_constraint=mask_3())
        ])

    self.G_DEC = Sequential([
        InputLayer(shape=(latent_dim2+1,)),
        Dense(units = latent_dim2+1, use_bias=False,kernel_initializer='zeros',activation="linear",kernel_constraint=mask_0()),
        Dense(units = (latent_dim2+1)*d2, use_bias=False,kernel_initializer=k2,activation="elu",kernel_constraint=mask_1()),
        Dense(units = (latent_dim2+1)*d2, use_bias=False,kernel_initializer=k2,activation="elu",kernel_constraint=mask_2()),
        Dense(units=latent_dim2+1,use_bias=False,kernel_initializer=k2,activation="linear",kernel_constraint=mask_3())
        ])





  #@tf.function

  def enc(self, x): #returns encoded noise eps
    ec = self.Enc(x,training=True)
    mean, lv = tf.split(ec,num_or_size_splits=2,axis=1)
    return mean, lv #lv: log-variance


  def reparam(self, mean, lv):
    eps = tf.random.normal(shape=mean.shape) #seed not fixed for randomness
    return eps*tf.math.exp(lv*0.5) + mean #lv: log variance

  def lat_split(self,x):
    lat1, lat2 = tf.split(x, [self.latent_dim1, self.latent_dim2],axis=1)
    return lat1, lat2

  def dec(self,z, sigmoid=False):
    result = self.Dec(z,training=True)
    if sigmoid == True:
        result = tf.math.sigmoid(result)
        return result
    return result


  mse_loss = tf.keras.losses.MeanSquaredError()
  def gae_loss(self,x,y): #y used for weak supervision
    mean, lv = self.enc(x)
    z = self.reparam(mean, lv)
    z1,z2 = self.lat_split(z)
    z2 = tf.concat([z2,y.reshape(-1,1)],axis=1)
    z2_ = self.G_ENC(z2,training=True)
    z2_hat = self.G_DEC(z2_,training=True)
    return mse_loss(z2,z2_hat)



  b_loss2 = tf.keras.losses.BinaryCrossentropy()
  def C_ELBO_loss(model, x): #ELBO loss
    mean, lv = model.enc(x)
    z = model.reparam(mean, lv)
    z1,z2 = model.lat_split(z)
    z_ = tf.concat([z1,z2],axis=1)
    x_hat = model.dec(z_,sigmoid=False)
    flat = tf.keras.layers.Flatten()
    f_x_hat = flat(x_hat)
    f_x = flat(x)
    loss_2 = tf.norm(z, ord=2)
    return b_loss2(f_x,f_x_hat)+0.001*loss_2/2 #weighted, 0.001


  def distance_f(self,x):
    eps=1e-10
    r = tf.reduce_sum(x * x, axis=1, keepdims=True)
    D2 = r - 2 * tf.matmul(x, x, transpose_b=True) + tf.transpose(r)

    D2 = tf.maximum(D2, 0.0)

    euc_mat = tf.sqrt(D2 + eps)
    euc_k = tf.reduce_mean(euc_mat,axis=1)
    euc_l = tf.reduce_mean(euc_mat, axis=0)
    euc_mean = tf.reduce_mean(euc_mat)
    A_mat = euc_mat-tf.broadcast_to(euc_k[:,tf.newaxis],[x.shape[0],x.shape[0]])-tf.broadcast_to(euc_l[tf.newaxis,:],[x.shape[0],x.shape[0]])+euc_mean
    return A_mat

  def dCor(self,x,y):
    x = tf.cast(x,tf.float32)
    y = tf.cast(y,tf.float32)
    if len(y.shape) == 1:
        y = tf.expand_dims(y, axis=1)
    V2_x = tf.reduce_sum(self.distance_f(x)**2)/tf.cast((tf.shape(x)[0]**2),tf.float32)
    V2_y = tf.reduce_sum(self.distance_f(y)**2)/tf.cast((tf.shape(y)[0]**2),tf.float32)
    V2_x = tf.maximum(V2_x, 0.0)
    V2_y = tf.maximum(V2_y, 0.0)
    V2_xy = tf.reduce_sum(tf.multiply(self.distance_f(x),self.distance_f(y)))/tf.cast((tf.shape(y)[0]**2),tf.float32)
    V2_xy = tf.maximum(V2_xy, 0.0)
    R = tf.sqrt(V2_xy)/tf.sqrt(tf.sqrt(V2_x*V2_y)+1e-8)
    return R

  def entropy_(self,x,bins=5):
    x = tf.cast(x,tf.float32)
    x = tf.image.rgb_to_grayscale(x)
    x = tf.squeeze(x,axis=-1)
    x = tf.clip_by_value(x, 0.0, 1.0)

    x_bin = tf.cast(tf.floor(x * bins), tf.int32)
    x_bin = tf.clip_by_value(x_bin, 0, bins - 1)

    hist = tf.one_hot(x_bin, depth=bins)
    hist = tf.reduce_sum(hist, axis=[1, 2])

    prob = hist/tf.reduce_sum(hist,axis=1,keepdims=True)

    ent = -tf.reduce_sum(prob*tf.math.log(prob+1e-10)/tf.math.log(2.0),axis=1)
    return ent

  def var_loss(model,x):
    mean,lv = model.enc(x)
    z = model.reparam(mean,lv)
    loss = 0
    for i in range(model.latent_dim2):
      if i < (model.latent_dim2-1):
        z_ = z+tf.concat([tf.zeros([model.latent_dim1+i]),[tf.math.reduce_std(z[:,model.latent_dim1+i])],tf.zeros([model.latent_dim2-1-i])],axis=0)
      else:
        z_ = z+tf.concat([tf.zeros([model.latent_dim1+i]),[tf.math.reduce_std(z[:,model.latent_dim1+i])]],axis=0)
      x_hat = model.dec(z_)
      diff = tf.math.square(x-x_hat)
      disp = model.entropy_(diff)
      loss += tf.reduce_mean(disp)
    return loss/model.latent_dim2


  def dis_loss1(model,x,y,b):
    mean,lv = model.enc(x)
    z = model.reparam(mean,lv)
    z1, z2 = model.lat_split(z)
    dcor_z1=0; dcor_z2=0
    for i in range(z1.shape[1]):
      dcor_z1 += model.dCor(tf.expand_dims(z1[:,i], axis=1),y)
    for j in range(z2.shape[1]):
      dcor_z2 += model.dCor(tf.expand_dims(z2[:,j], axis=1),y)
    loss_total = dcor_z1/z1.shape[1]-b*dcor_z2/z2.shape[1]
    return loss_total

  def dis_loss2(model,x,y):
    mean,lv = model.enc(x)
    z = model.reparam(mean,lv)
    z1, z2 = model.lat_split(z)
    dcor_z12 = 0
    for j in range(z1.shape[1]-1):
      for i in range(j+1,z1.shape[1],1):
        dcor_z12 += model.dCor(tf.expand_dims(z1[:,j], axis=1),z1[:,i])
    dcor_z12 = dcor_z12/(z1.shape[1]*(z1.shape[1]-1)/2)
    return dcor_z12

################Defining Model

#Additional Random Sampling from train data for latent variation block
np.random.seed(321)
idx__ = np.random.choice([i for i in range(0,240,1)],50,replace=False)
X_tr3 = X_tr2[idx__,:,:,:]

#Updating using LungCRCT Framework

tf.keras.utils.set_random_seed(987)
os.environ['TF_DETERMINISTIC_OPS']='1'

tf.executing_eagerly()
import tensorflow.keras.backend as K

Epochs = 450
lat_dim_1 = 5
lat_dim_2 = 3
LungCRCT = lungcrct(180,lat_dim_1,lat_dim_2,16,32,32,16,128,16,4)
mse_loss = tf.keras.losses.MeanSquaredError()
b_loss2 = tf.keras.losses.BinaryCrossentropy()


alpha=0.6
i = 0
rho = 0.1
gamma=0.9
beta = 1.01
#lamb = 0.0 (L1-regularization not implemented)


loss_of_cs = []
loss_of_cv = []
loss_of_rc = []
loss_of_dcor = []
basic_opt0 = tf.keras.optimizers.Adam(learning_rate=0.007)
basic_opt = tf.keras.optimizers.Adam(learning_rate=0.002)
basic_opt2 = tf.keras.optimizers.Adam(learning_rate=0.002)
basic_opt3 = tf.keras.optimizers.Adam(learning_rate=0.001)
basic_opt4 = tf.keras.optimizers.Adam(learning_rate=0.0007)
loss_3 = 0

I = tf.eye(lat_dim_2+1,dtype=tf.float32)
s=1

while i < Epochs:
    with tf.GradientTape() as dv_t0, tf.GradientTape() as dv_t, tf.GradientTape() as dv_t2, tf.GradientTape() as dv_t3, tf.GradientTape() as dv_t4:
      loss_ = LungCRCT.C_ELBO_loss(X_tr2)
      loss_2 = LungCRCT.gae_loss(X_tr2,tag_tr)

      h_a = -tf.math.log(tf.linalg.det(s*I-tf.math.multiply(LungCRCT.G_DEC.weights[0], LungCRCT.G_DEC.weights[0])))+(lat_dim_2+1)*np.log(s)
      if i < 50:
        cs_l = loss_ + 0.5*(loss_2 + alpha*h_a+rho*0.5*tf.math.abs(h_a)**2)
      elif i< 100:
        loss_3 = LungCRCT.dis_loss1(X_tr2,tag_tr,1.5)
        loss_4 = LungCRCT.var_loss(X_tr3)
        cs_l = loss_ + loss_3 + loss_2 + alpha*h_a+rho*0.5*tf.math.abs(h_a)**2
      else:
        loss_3 = LungCRCT.dis_loss1(X_tr2,tag_tr,1.5)
        loss_4 = LungCRCT.var_loss(X_tr3)
        cs_l = loss_ + loss_3 + loss_2 + loss_4 + alpha*h_a+rho*0.5*tf.math.abs(h_a)**2
      d_l = loss_2 + alpha*h_a+rho*0.5*tf.math.abs(h_a)**2

    loss_of_cs.append(cs_l)
    loss_of_cv.append(d_l)
    loss_of_rc.append(loss_)
    loss_of_dcor.append(loss_3)
    grad_g_0 = dv_t0.gradient(d_l, [LungCRCT.G_DEC.trainable_variables[0]])
    grad_g_a = dv_t.gradient(d_l, LungCRCT.G_DEC.trainable_variables[1:])
    grad_g_b = dv_t2.gradient(d_l, LungCRCT.G_ENC.trainable_variables)
    grad_g_c = dv_t3.gradient(cs_l, LungCRCT.Enc.trainable_variables)
    grad_g_d = dv_t4.gradient(cs_l, LungCRCT.Dec.trainable_variables)
    basic_opt0.apply_gradients(zip(grad_g_0, [LungCRCT.G_DEC.trainable_variables[0]]))
    basic_opt.apply_gradients(zip(grad_g_a, LungCRCT.G_DEC.trainable_variables[1:]))
    basic_opt2.apply_gradients(zip(grad_g_b, LungCRCT.G_ENC.trainable_variables))
    basic_opt3.apply_gradients(zip(grad_g_c,LungCRCT.Enc.trainable_variables))
    basic_opt4.apply_gradients(zip(grad_g_d, LungCRCT.Dec.trainable_variables))
    h_a_new = -tf.math.log(tf.linalg.det(s*I-tf.math.multiply(LungCRCT.G_DEC.weights[0], LungCRCT.G_DEC.weights[0])))+(lat_dim_2+1)*np.log(s)
    alpha =  alpha + rho * h_a_new
    if (tf.math.abs(h_a_new) >= gamma*tf.math.abs(h_a)):
        rho = beta*rho
    else:
        rho = rho
    if (i+1) %10 == 0: print(i+1, d_l,loss_,loss_2,loss_3)
    i = i+1

###########################Causal Representation Learning Results

#causal weighted adjacency matrix(fitted)
import seaborn as sns
sns.set_style("darkgrid")
sns.heatmap(np.array(LungCRCT.G_DEC.weights[0]), cmap="vlag",center=0)
plt.show()

#binarized adjacency matrix
import seaborn as sns
sns.set_style("darkgrid")
sns.heatmap(np.where(np.abs(np.array(LungCRCT.G_DEC.weights[0]))> np.quantile(np.abs(np.array(LungCRCT.G_DEC.weights[0])),0.7),1,0), cmap="gray_r",linewidths=1,linecolor="black") #0.7
plt.show()

tf.keras.utils.set_random_seed(456)
mean, lv = LungCRCT.enc(X_tr2)
z = LungCRCT.reparam(mean, lv)
z1,z2 = LungCRCT.lat_split(z)
z_ = tf.concat([z1,z2],axis=1)
x_hat = LungCRCT.dec(z_, sigmoid=False)

#Mutual Information for each latent variable in latent space Z
from sklearn.feature_selection import mutual_info_regression
print(mutual_info_regression(np.array(z1[:,0]).reshape(-1,1),tag_tr,n_neighbors=5,random_state=321))
print(mutual_info_regression(np.array(z1[:,1]).reshape(-1,1),tag_tr,n_neighbors=5,random_state=321))
print(mutual_info_regression(np.array(z1[:,2]).reshape(-1,1),tag_tr,n_neighbors=5,random_state=321))
print(mutual_info_regression(np.array(z1[:,3]).reshape(-1,1),tag_tr,n_neighbors=5,random_state=321))
print(mutual_info_regression(np.array(z1[:,4]).reshape(-1,1),tag_tr,n_neighbors=5,random_state=321))
print(mutual_info_regression(np.array(z2[:,0]).reshape(-1,1),tag_tr,n_neighbors=5,random_state=321))
print(mutual_info_regression(np.array(z2[:,1]).reshape(-1,1),tag_tr,n_neighbors=5,random_state=321))
print(mutual_info_regression(np.array(z2[:,2]).reshape(-1,1),tag_tr,n_neighbors=5,random_state=321))

import seaborn as sns
sns.set_style("darkgrid")
plt.plot(loss_of_cv, color="black",label="Loss_2")
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()

import seaborn as sns
sns.set_style("darkgrid")
plt.plot(loss_of_cs, color="red",label="Total_loss")
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()

import seaborn as sns
sns.set_style("darkgrid")
plt.plot(loss_of_rc, color="darkblue",label="Loss_1")
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()

import seaborn as sns
sns.set_style("darkgrid")
plt.plot(loss_of_dcor, color="darkgreen",label="Loss_3")
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()

d_l #0.005920679774135351

loss_of_cs[-1] #-0.5390983819961548

print(loss_,loss_2,loss_3,loss_4)

blended_img = cv2.addWeighted(np.array(x_hat[30,:,:,0],dtype=np.float32),0.8, np.array(X_tr2[30,:,:,0],dtype=np.float32), 0.2, 0)

plt.imshow(blended_img)
plt.grid()

#TSNE results for LungCRCT fitting

from sklearn.preprocessing import StandardScaler, MinMaxScaler

st1 = StandardScaler()

z2_tr = st1.fit_transform(z2)

from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, perplexity=20, learning_rate=100, random_state=123)

X_tsne = tsne.fit_transform(z2_tr)

tsne_df = pd.DataFrame(data=X_tsne, columns=['TSNE1', 'TSNE2'])
tsne_df['Target'] = tag_tr

sns.scatterplot(x='TSNE1',y='TSNE2',hue='Target',palette="dark",data=tsne_df,legend='full',alpha=0.9)
plt.title('t-SNE Visualization of Z_cs space')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')

st2 = StandardScaler()

z1_tr = st2.fit_transform(z1)

tsne1 = TSNE(n_components=2, perplexity=20, learning_rate=100, random_state=123)

X_tsne1 = tsne1.fit_transform(z1_tr)

tsne_df1 = pd.DataFrame(data=X_tsne1, columns=['TSNE1', 'TSNE2'])
tsne_df1['Target'] = tag_tr

sns.scatterplot(x='TSNE1',y='TSNE2',hue='Target',palette="dark",data=tsne_df1,legend='full',alpha=0.9)
plt.title('t-SNE Visualization of Z_ncs space')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')



####################Qualitative Checks for disentanglement

tf.keras.utils.set_random_seed(321321)
mean, lv = LungCRCT.enc(X_tr2)
z = LungCRCT.reparam(mean, lv)
z1,z2 = LungCRCT.lat_split(z)
z_ = tf.concat([z1,z2],axis=1)

#Min-Max range for each latent causal variables
print(np.min(z_[:,5]),np.max(z_[:,5]))
print(np.min(z_[:,6]),np.max(z_[:,6]))
print(np.min(z_[:,7]),np.max(z_[:,7]))

#Disentanglement simulation (Qualitative) Z_cs,0
set1 = [-0.8,-0.6,-0.4,-0.2,0]

fig3 = plt.figure(figsize=(14,3))
rows = 1; columns=5
ax3 = []

for i in range(5):
  distangle = np.array(z_[72,:]) #32,44,72,90
  distangle[5] =  set1[4-i]
  distg = tf.reshape(tf.convert_to_tensor(distangle, dtype=tf.float32),[1,8])
  distg = LungCRCT.dec(distg, sigmoid=False)
  ax3.append(fig3.add_subplot(rows,columns,i+1))
  ax3[-1].set_title("Z_cs,0:"+str(set1[4-i]))
  plt.axis("off")
  plt.imshow(distg[0,:,:,:])

#Visualized Results
plt.tight_layout(pad=0.00)
plt.subplots_adjust(top = 1, bottom = 0, right = 2, left = 1.1, hspace = 0.5, wspace = 0)
plt.show()

######

#Disentanglement simulation (Qualitative) Z_cs,1
set1 = [-0.7,-0.5,-0.3,-0.1,0.1]

fig3 = plt.figure(figsize=(14,3))
rows = 1; columns=5
ax3 = []

for i in range(5):
  distangle = np.array(z_[44,:]) #32,44,72,90
  distangle[6] = set1[4-i]
  distg = tf.reshape(tf.convert_to_tensor(distangle, dtype=tf.float32),[1,8])
  distg = LungCRCT.dec(distg, sigmoid=False)
  ax3.append(fig3.add_subplot(rows,columns,i+1))
  ax3[-1].set_title("Z_cs,1:"+str(set1[4-i]))
  plt.axis("off")
  plt.imshow(distg[0,:,:,:])

#Visualized Results
plt.tight_layout(pad=0.00)
plt.subplots_adjust(top = 1, bottom = 0, right = 2, left = 1.1, hspace = 0.5, wspace = 0)
plt.show()

#####

#Disentanglement simulation (Qualitative) Z_cs,2
set1 = [-0.7,-0.5,-0.3,-0.1,0.1]

fig3 = plt.figure(figsize=(14,3))
rows = 1; columns=5
ax3 = []

for i in range(5):
  distangle = np.array(z_[32,:]) ##32,44,72,90
  distangle[7] = set1[4-i]
  distg = tf.reshape(tf.convert_to_tensor(distangle, dtype=tf.float32),[1,8])
  distg = LungCRCT.dec(distg, sigmoid=False)
  ax3.append(fig3.add_subplot(rows,columns,i+1))
  ax3[-1].set_title("Z_cs,2:"+str(set1[4-i]))
  plt.axis("off")
  plt.imshow(distg[0,:,:,:])

#Visualized Results
plt.tight_layout(pad=0.00)
plt.subplots_adjust(top = 1, bottom = 0, right = 2, left = 1.1, hspace = 0.5, wspace = 0)
plt.show()

###################

k1 = 90
plt.imshow(X_tr2[k1,:,:,:])#Original image. 32, 44, 72, 90
print(tag_tr[k1])
plt.grid()
plt.show()

################################################## Downstream Task



#Downstream task experiments: encoding train/test data for analysis based on LungCRCT encoder.
#Train data encodings
tf.keras.utils.set_random_seed(1111)
mean1, lv1 = LungCRCT.enc(X_tr2)
z_tr = LungCRCT.reparam(mean1, lv1)
z_tr1,z_tr2 = LungCRCT.lat_split(z_tr)
z_tr = tf.concat([z_tr1,z_tr2],axis=1)

#Test data encodings
tf.keras.utils.set_random_seed(1111)
mean2, lv2 = LungCRCT.enc(X_te2)
z_te = LungCRCT.reparam(mean2, lv2)
z_te1,z_te2 = LungCRCT.lat_split(z_te)
z_te = tf.concat([z_te1,z_te2],axis=1)



#Converting label Y to binary status
tag_tr2 = np.where(tag_tr>1,1,0)
tag_te2 = np.where(tag_te>1,1,0)

#Only using latent space Z_cs for downstream tasks.
z_tr_ = np.array(z_tr2)
z_te_ = np.array(z_te2)

from sklearn.metrics import roc_auc_score

############

LungCRCT.Enc.summary()

#DNN for predicting LungCancer Status(Malignant/non-Malignant)
tf.keras.utils.set_random_seed(321321)
os.environ['TF_DETERMINISTIC_OPS']='1'
k3 = tf.keras.initializers.HeNormal(123)

md_d = Sequential([
    InputLayer(shape=(3,)),
    BatchNormalization(),
    Dense(units = 32, kernel_initializer=k3, activation="elu"),
    Dense(units = 32, kernel_initializer=k3, activation="elu"),
    Dense(units=1,kernel_initializer=k3, activation="sigmoid")
    ])

md_d.summary()

from tensorflow.keras.optimizers import RMSprop, Adam
md_d.compile(optimizer=Adam(learning_rate=1e-4), loss=tf.keras.losses.BinaryCrossentropy,metrics=['accuracy']) #1e-4

hist3 = md_d.fit(x=z_tr_, y=tag_tr2,epochs=300,batch_size=80,shuffle=True) #

import seaborn as sns
sns.set_style("darkgrid")
plt.plot(hist3.history['loss'], color="red",label="loss")

f1_score(tag_tr2,np.round(md_d.predict(z_tr_)),average="macro")

prd_ = md_d.predict(z_tr_)
prd_tr = []
for i in range(len(tag_tr2)):
  if prd_[i] < 0.5:
    prd_tr.append(0)
  else:
    prd_tr.append(1)

from sklearn.metrics import classification_report
print(classification_report(tag_tr2,prd_tr,digits=4))

###Test data performance

prd = md_d.predict(z_te_)
prd1 = []
for i in range(len(tag_te2)):
  if prd[i] < 0.5:
    prd1.append(0)
  else:
    prd1.append(1)

from sklearn.metrics import ConfusionMatrixDisplay
dis = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(tag_te2,prd1), display_labels=[0,1])
dis.plot(cmap="Blues")
plt.grid()
plt.show()

from sklearn.metrics import classification_report
print(classification_report(tag_te2,prd1,digits=4))

from sklearn.metrics import precision_score, recall_score,f1_score
print(recall_score(tag_te2,prd1,average="macro"))
print(precision_score(tag_te2,prd1,average="macro"))
print(f1_score(tag_te2,prd1,average="macro"))

from sklearn.metrics import roc_auc_score

roc_auc_score(tag_te2,md_d.predict(z_te_))

from sklearn.metrics import roc_auc_score, roc_curve

fpr1, tpr1, _1 = roc_curve(tag_te2, md_d.predict(z_te_))
plt.plot(fpr1,tpr1, label="LungCRCT-based Approach"+" area:"+str(np.round(roc_auc_score(tag_te2,md_d.predict(z_te_)),4)),color="darkcyan")
plt.plot(np.linspace(0,1,1000),np.linspace(0,1,1000),color="black",linestyle="-.")
plt.legend()
#plt.grid()
plt.show()
