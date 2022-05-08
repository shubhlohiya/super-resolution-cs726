import tensorflow as tf
from matplotlib import pyplot as plt
import os
import time
import cv2
import numpy as np
from tensorflow.keras.layers import Add,Concatenate,LeakyReLU,Conv2D,Lambda,UpSampling2D
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from keras import backend as K
import tensorflow as tf
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Activation, Add, Concatenate, Multiply
from keras.layers import BatchNormalization, LeakyReLU, PReLU, Conv2D, Dense
from keras.layers import UpSampling2D, Lambda, Dropout
from keras.applications.vgg19 import preprocess_input
from keras.utils.data_utils import OrderedEnqueuer
from keras import backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint, LambdaCallback
import os
import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPool2D, Activation, AveragePooling2D, MaxPool2D, Dense, Conv2D,Lambda

class Channel_Attention(tf.keras.layers.Layer) : # Channel attention module assuming the input dimensions to have channels-last order
  def __init__(self,C,ratio) :
    super(Channel_Attention,self).__init__()
    self.avg_pool = GlobalAveragePooling2D()
    self.max_pool = GlobalMaxPool2D()
    self.activation = Activation('sigmoid')
    self.fc1 = Dense(C/ratio, activation = 'relu')
    self.fc2 = Dense(C)
  
  def call(self,x) :
    avg_out1 = self.avg_pool(x)
    avg_out2 = self.fc1(avg_out1)
    avg_out3 = self.fc2(avg_out2)
    max_out1 = self.max_pool(x)
    max_out2 = self.fc1(max_out1)
    max_out3 = self.fc2(max_out2)
    add_out = tf.math.add(max_out3,avg_out3)
    channel_att = self.activation(add_out)
    return channel_att  

class Spatial_Attention(tf.keras.layers.Layer) : # spatial attention module assuming the input dimensions to have channels-last order
  def __init__(self) :
    super(Spatial_Attention,self).__init__()
    self.conv2d = Conv2D(1,(7,7),padding='same',activation='sigmoid')
    self.avg_pool_chl = Lambda(lambda x:tf.keras.backend.mean(x,axis=3,keepdims=True)) # avg-pooling along channel axis
    self.max_pool_chl = Lambda(lambda x:tf.keras.backend.max(x,axis=3,keepdims=True))  # max-pooling along channel axis
  
  def call(self,x) :
    avg_out1 = self.avg_pool_chl(x)
    max_out1 = self.max_pool_chl(x)
    concat_out = tf.concat([avg_out1,max_out1],axis=-1)
    spatial_att = self.conv2d(concat_out)
    return spatial_att 

class CBAM(tf.keras.layers.Layer) : # convolutional block attention module assuming the input dimensions to have channels-last order 
  def __init__(self,C,ratio) :
    super(CBAM,self).__init__()
    self.C = C
    self.ratio = ratio
    self.channel_attention = Channel_Attention(self.C,self.ratio)
    self.spatial_attention = Spatial_Attention()
  def call(self,y,H,W,C) :
    ch_out1 = self.channel_attention(y)
    ch_out2 = tf.expand_dims(ch_out1, axis=1)
    ch_out3 = tf.expand_dims(ch_out2, axis=2)
    ch_out4 = tf.tile(ch_out3, multiples=[1,H,W,1])
    ch_out5 = tf.math.multiply(ch_out4,y)
    sp_out1 = self.spatial_attention(ch_out5)
    sp_out2 = tf.tile(sp_out1, multiples = [1,1,1,C])
    sp_out3 = tf.math.multiply(sp_out2,ch_out5)
    return sp_out3        

cbam1 = CBAM(64,4)
cbam2 = CBAM(64,4)
cbam3 = CBAM(64,4)

def random_crop(input_image):
    start_height = np.random.randint(0,input_image.shape[0]-96)
    start_width = np.random.randint(0,input_image.shape[1]-96)
    image = input_image[start_height:start_height+96 , start_width:start_width+96]

    return image

def load_hr(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_png(image)
    image = np.asarray(image)
    hr_image = random_crop(image)

    return hr_image

def load_lr(hr_image):
    lr_image = cv2.blur(hr_image,(3,3))
    lr_image = cv2.resize(lr_image, (24,24))
    return lr_image

def normalize(image):
    image_t = tf.convert_to_tensor(image , dtype = tf.float32)
    image_t = image_t/127.5 -1
    return image_t

PATH = "data/DIV2K_train_HR"
train_dataset = os.listdir(PATH)
for i in range(len(train_dataset)):
  train_dataset[i] = PATH + '/'+train_dataset[i]
train_hr_dataset = list(map(load_hr, train_dataset))
train_lr_dataset = list(map(load_lr,train_hr_dataset))
train_hr_dataset = tf.convert_to_tensor(list(map(normalize , train_hr_dataset)))
train_lr_dataset = tf.convert_to_tensor(list(map(normalize , train_lr_dataset)))

upscaling_factor = 4

def buildGenerator():
        """
        Build the generator network according to description in the paper.
        :return: the compiled model
        """
        def SubpixelConv2D(name, scale=2):

            def subpixel_shape(input_shape):
                dims = [input_shape[0],
                    None if input_shape[1] is None else input_shape[1] * scale,
                    None if input_shape[2] is None else input_shape[2] * scale,
                    int(input_shape[3] / (scale ** 2))]
                output_shape = tuple(dims)
                return output_shape

            def subpixel(x):
                return tf.nn.depth_to_space(x, scale)

            return Lambda(subpixel, output_shape=subpixel_shape, name=name)

        def dense_block(input):
            x1 = Conv2D(64, kernel_size=3, strides=1, padding='same')(input)
            x1 = LeakyReLU(0.2)(x1)
            x1 = Concatenate()([input, x1])

            x2 = Conv2D(64, kernel_size=3, strides=1, padding='same')(x1)
            x2 = LeakyReLU(0.2)(x2)
            x2 = Concatenate()([input, x1, x2])

            x3 = Conv2D(64, kernel_size=3, strides=1, padding='same')(x2)
            x3 = LeakyReLU(0.2)(x3)
            x3 = Concatenate()([input, x1, x2, x3])

            x4 = Conv2D(64, kernel_size=3, strides=1, padding='same')(x3)
            x4 = LeakyReLU(0.2)(x4)
            x4 = Concatenate()([input, x1, x2, x3, x4])  # 这里跟论文原图有冲突，论文没x3???

            x5 = Conv2D(64, kernel_size=3, strides=1, padding='same')(x4)
            x5 = Lambda(lambda x: x * 0.2)(x5)
            x = Add()([x5, input])
            return x

        def RRDB(input):
            x = dense_block(input)
            x = cbam1(x,x.shape[1],x.shape[2],x.shape[3])
            x = dense_block(x)
            x = cbam2(x,x.shape[1],x.shape[2],x.shape[3])
            x = dense_block(x)
            x = cbam3(x,x.shape[1],x.shape[2],x.shape[3])
            x = Lambda(lambda x: x * 0.2)(x)
            out = Add()([x, input])
            return out

        def upsample(x, number):
            x = Conv2D(256, kernel_size=3, strides=1, padding='same', name='upSampleConv2d_' + str(number))(x)
            x = SubpixelConv2D('upSampleSubPixel_' + str(number), 2)(x)
            x = PReLU(shared_axes=[1, 2], name='upSamplePReLU_' + str(number))(x)
            return x

        # Input low resolution image
        lr_input = Input(shape=(24, 24, 3))

        # Pre-residual
        x_start = Conv2D(64, kernel_size=3, strides=1, padding='same')(lr_input)
        x_start = LeakyReLU(0.2)(x_start)

        # Residual-in-Residual Dense Block
        x = RRDB(x_start)

        # Post-residual block
        x = Conv2D(64, kernel_size=3, strides=1, padding='same')(x)
        x = Lambda(lambda x: x * 0.2)(x)
        x = Add()([x, x_start])

        # Upsampling depending on factor
        x = upsample(x, 1)
        if upscaling_factor > 2:
            x = upsample(x, 2)
        if upscaling_factor > 4:
            x = upsample(x, 3)

        x = Conv2D(64, kernel_size=3, strides=1, padding='same')(x)
        x = LeakyReLU(0.2)(x)
        hr_output = Conv2D(3, kernel_size=3, strides=1, padding='same', activation='tanh')(x)

        # Create model and compile
        model = Model(inputs=lr_input, outputs=hr_output)
        # model.summary()
        return model

generator = buildGenerator()

def build_discriminator():
    leakyrelu_alpha = 0.2
    momentum = 0.8

    input_0 = tf.keras.layers.Input(shape=(24,24,3))
    input_0_upscale = UpSampling2D(4)(input_0)

    input_1 = tf.keras.layers.Input(shape=(96,96,3))
    input_2 = tf.keras.layers.Input(shape = (96,96,3))

    x = tf.keras.layers.concatenate([input_0_upscale,input_1])
    y = tf.keras.layers.concatenate([input_0_upscale,input_2])
    for i in range(4):
      x = Conv2D(64 , kernel_size = 6 , strides = 1 , padding = 'same')(x)
      y = Conv2D(64 , kernel_size = 6 , strides = 1 , padding = 'same')(y)
      x = LeakyReLU()(x)
      y = LeakyReLU()(y)
      x = tf.keras.layers.BatchNormalization()(x)
      y = tf.keras.layers.BatchNormalization()(y)

    logits = x-K.mean(y)
    # fully connected layer 
    output = Conv2D(1,4, activation='sigmoid' , padding = 'same')(logits)   
    
    model = tf.keras.Model(inputs=[input_0 , input_1,input_2], outputs=[output], name='discriminator')
    
    return model

discriminator = build_discriminator()

def relativistic_loss(disc_real,disc_gen):
    real = disc_real
    fake = disc_gen
    fake_logits = K.sigmoid(fake - K.mean(real))
    real_logits = K.sigmoid(real - K.mean(fake))
            
    return [fake_logits, real_logits]

def discriminator_loss(fake_logits , real_logits) :
  return  K.mean(K.binary_crossentropy(K.zeros_like(fake_logits),fake_logits)+K.binary_crossentropy(K.ones_like(real_logits),real_logits))

def generator_loss(fake_logits , real_logits) :
  return  K.mean(K.binary_crossentropy(K.zeros_like(real_logits),real_logits)+K.binary_crossentropy(K.ones_like(fake_logits),fake_logits))

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)


# The VGG model for VGG loss , made for our input shape 
vgg = VGG19(include_top = False, input_shape=(96,96,3))

Lambda = 0.05
Eeta = 1 #both these values are supposed to be changed after epochs. The initial values are such that the GAN first predicts a rough figure about the images
EPOCHS = 2000


def train_step(input_lr_image, target, epoch):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_lr_image, training=True)

        real_logits  = discriminator([input_lr_image, target , gen_output], training=True)
        fake_logits  = discriminator([input_lr_image,  gen_output, target], training=True)
        gen_loss     = Lambda*generator_loss(fake_logits, real_logits)
        gen_loss    += Eeta*tf.reduce_mean(tf.abs(target - gen_output))
        feature_gen  = vgg(preprocess_input(gen_output))
        feature_real = vgg(preprocess_input(np.copy(target)))
        vgg_loss     = tf.keras.losses.mean_squared_error(feature_gen , feature_real)
        gen_loss    += 100*vgg_loss
        gen_loss = tf.reduce_mean(gen_loss)
        disc_loss = 5*discriminator_loss(fake_logits, real_logits)

    generator_gradients = gen_tape.gradient(gen_loss,
                                          generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))
    return gen_loss, disc_loss

def fit(train_lr,train_hr, epochs):
    for epoch in range(epochs):
        start = time.time()
        print(".")
        for i in range(200):
          input_image  = train_lr[4*i:4*i+4]
          target  =  train_hr[4*i:4*i+4]
          gen_loss, disc_loss = train_step(input_image,target , epoch)
        print('Generator Loss', gen_loss, 'Discriminator Loss', disc_loss)
        if (epoch + 1) % 100 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)
        generated = generator(train_lr_dataset[0:5])
        print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                        time.time()-start))

checkpoint_dir = '/mnt/dog/data/shubhlohiya/cs726/esrgan_cbam_updated_ckpts'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,discriminator_optimizer=discriminator_optimizer,generator=generator,discriminator=discriminator)

fit(train_lr_dataset,train_hr_dataset, EPOCHS)

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


