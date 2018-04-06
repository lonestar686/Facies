
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout
from keras.layers import BatchNormalization, Activation
from keras.layers.merge import concatenate

from nets.MyNet import myNet

class Unet_BN(myNet):
    """ Unet with batch normalization
    """
    def __init__(self, img_rows = 512, img_cols = 512, img_nchs = 1, nclasses=1, \
		         out_dir='./results', model_dir='./model', **kwargs):
        super(Unet_BN, self).__init__(img_rows, img_cols, img_nchs, nclasses, \
		                          out_dir, model_dir, **kwargs)

    def get_net(self):

        print("using net {}".format(__name__))

        #
        input_shape = (self.img_rows, self.img_cols, self.img_nchs)
        nclasses = self.nclasses

        #
        img_input = Input(shape=input_shape)
        
    # first convolution
        conv1 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(img_input)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation('relu')(conv1)
        #
        conv1 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(conv1)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation('relu')(conv1)
        #
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        print("pool1 shape:",pool1.shape)

    # second convolution
        conv2 = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(pool1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation('relu')(conv2)
        #
        conv2 = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(conv2)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation('relu')(conv2)
        #
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        print("pool2 shape:",pool2.shape)

    # third convolution
        conv3 = Conv2D(256, 3, padding = 'same', kernel_initializer = 'he_normal')(pool2)
        conv3 = BatchNormalization()(conv3)
        conv3 = Activation('relu')(conv3)
        #
        conv3 = Conv2D(256, 3, padding = 'same', kernel_initializer = 'he_normal')(conv3)
        conv3 = BatchNormalization()(conv3)
        conv3 = Activation('relu')(conv3)
        #
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        print("pool3 shape:",pool3.shape)

    # fourth convolution
        conv4 = Conv2D(512, 3, padding = 'same', kernel_initializer = 'he_normal')(pool3)
        conv4 = BatchNormalization()(conv4)
        conv4 = Activation('relu')(conv4)
        #    
        conv4 = Conv2D(512, 3, padding = 'same', kernel_initializer = 'he_normal')(conv4)
        conv4 = BatchNormalization()(conv4)
        conv4 = Activation('relu')(conv4)
        #
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    # fifth convolution
        conv5 = Conv2D(1024, 3, padding = 'same', kernel_initializer = 'he_normal')(pool4)
        conv5 = BatchNormalization()(conv5)
        conv5 = Activation('relu')(conv5)
        #
        conv5 = Conv2D(1024, 3, padding = 'same', kernel_initializer = 'he_normal')(conv5)
        conv5 = BatchNormalization()(conv5)
        conv5 = Activation('relu')(conv5)

        drop5 = Dropout(0.5)(conv5)

    # upward
        up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
        merge6 = concatenate([drop4,up6], axis = 3)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

        up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
        merge7 = concatenate([conv3,up7], axis = 3)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

        up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
        merge8 = concatenate([conv2,up8], axis = 3)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

        up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
        merge9 = concatenate([conv1,up9], axis = 3)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

    #
        conv10 = Conv2D(nclasses, 1, activation = 'sigmoid')(conv9)

        model = Model(inputs = img_input, outputs = conv10)

        return model

