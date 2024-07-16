import numpy as np
import tensorflow as tf
print (np.version.version)
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

num_classes = 14
wspan= 100
hspan = 100
nhood = 1

X_train = np.load('X_test.npy')
Y_train = np.load('Y_test.npy')

Y_train = Y_train.reshape(len(Y_train),100,100,1)

tf.random.set_seed(0)
layer_dims = [100, 100, 100]
loss = lambda x, y : tf.keras.losses.categorical_crossentropy(tf.reshape(x, shape=(-1, num_classes)), 
                                                              tf.reshape(y, shape=(-1, num_classes)), 
                                                              from_logits=True)
diameter = 2*nhood+1
model = tf.keras.Sequential()
model.add(tf.keras.layers.InputLayer((wspan, hspan, 1)))
model.add(tf.keras.layers.Conv2D(100, kernel_size=[diameter, diameter], padding='same', 
                                 activation='relu', kernel_initializer=tf.keras.initializers.he_normal(), 
                                 bias_initializer=tf.keras.initializers.he_normal()))
for i in range(11):
    model.add(tf.keras.layers.Conv2D(100, kernel_size=[1,1], padding='same', 
                                 activation='relu', kernel_initializer=tf.keras.initializers.he_normal(), 
                                 bias_initializer=tf.keras.initializers.he_normal()))
model.add(tf.keras.layers.Conv2D(1, kernel_size=[1,1], padding='same', 
                                 activation='relu', kernel_initializer=tf.keras.initializers.he_normal(), 
                                 bias_initializer=tf.keras.initializers.he_normal()))

#model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3), loss=loss,metrics=['accuracy'])
model.compile(optimizer=tf.keras.optimizers.SGD(lr=1e-1, nesterov=True), loss=loss,metrics=['accuracy'])

EPOCHS = 150
checkpoint_filepath = 'best_working_SGD_nesterov_e-1_keras_model_2.h5'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)
print(model.summary())



train_history = model.fit(x=X_train, y=Y_train, epochs=EPOCHS, batch_size=28,shuffle = True,verbose=1,validation_split=0.2,callbacks=[model_checkpoint_callback])


