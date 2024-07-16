#!/usr/bin/env python
# coding: utf-8

# ## Import requirements

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import tensorflow as tf
import matplotlib.animation as anim

# In[3]:


num_classes = 14
wspan= 100
hspan = 100
nhood = 1


# # load data and model

# In[4]:


data = np.load('../../data/i0-0.5.npy')


# In[5]:


#### Define and build model
tf.random.set_seed(0)
layer_dims = [100, 100, 100]
loss = lambda x, y : tf.keras.losses.categorical_crossentropy(tf.reshape(x, shape=(-1, num_classes)), 
                                                              tf.reshape(y, shape=(-1, num_classes)), 
                                                              from_logits=True)
diameter = 2*nhood+1
model = tf.keras.Sequential()
model.add(tf.keras.layers.InputLayer((wspan, hspan, 1)))
model.add(tf.keras.layers.Conv2D(layer_dims[0], kernel_size=[diameter, diameter], padding='same', 
                                 activation='relu', kernel_initializer=tf.keras.initializers.he_normal(), 
                                 bias_initializer=tf.keras.initializers.he_normal()))
for i in range(1, len(layer_dims)):
    model.add(tf.keras.layers.Dense(layer_dims[i],  activation='relu',
                                    kernel_initializer=tf.keras.initializers.he_normal(), 
                                    bias_initializer=tf.keras.initializers.he_normal()))
model.add(tf.keras.layers.Dense(num_classes,  activation='relu',
                                kernel_initializer=tf.keras.initializers.he_normal(), 
                                bias_initializer=tf.keras.initializers.he_normal()))

#model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3), loss=loss,metrics=['accuracy'])
model.compile(optimizer=tf.keras.optimizers.SGD(lr=1e-1, nesterov=True), loss=loss,metrics=['accuracy'])

EPOCHS = 150
checkpoint_filepath = 'best_working_SGD_nesterov_e-1_keras_temp.h5'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)
model.summary()


# In[6]:


###  Load model
file='best_working_SGD_nesterov_e-1_keras.h5'
model.load_weights(file)


# In[7]:


data.shape


# In[29]:


temp = 3
output_array = [data[temp%500][0]] 
output = data[temp%500][0].reshape(1,100,100,1)
for i in range(29):
    labels = tf.argmax(tf.nn.softmax(model(output)), axis=-1)
    output = tf.reshape(labels, (-1, wspan, hspan,1))
    output_array.append(tf.reshape(labels, (wspan, hspan)))

N = 100
trans=15;obs=15;T=trans+obs

tau_i = 4 
tau_r = 9
tau_0 = tau_i+tau_r

##Define discreet colormap
cmap = colors.ListedColormap(['xkcd:pale grey','xkcd:darkish red','xkcd:almost black'])
bounds = [0,0.99,tau_i+0.99,tau_0+0.99]
norm = colors.BoundaryNorm(bounds,cmap.N)

def animate(i):
    grid.set_data(output_array[i]) 
    ax.set_title("$t={}$".format(i+1))
    return grid,

fig,ax = plt.subplots()
ax.set_xticks([])
ax.set_yticks([])
ax.set_aspect('equal')
grid=ax.imshow(output_array[0],origin='lower',cmap=cmap,norm=norm,interpolation='none')
ani = anim.FuncAnimation(fig, animate, T, interval=100,repeat=False)
plt.show()

count=0
for i in range(hspan):
    for j in range(wspan):
        if data[temp][29][i][j]!=output_array[29][i][j]:
            count+=1
print("accuracy is ",count/(wspan*hspan))
