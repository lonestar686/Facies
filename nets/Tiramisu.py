
from keras.layers import Input, Activation, Dropout, BatchNormalization
from keras.layers import concatenate, Reshape
from keras.layers import Conv2D, Conv2DTranspose
from keras.regularizers import l2
from keras.models import Model

from nets.MyNet import myNet

# helper functions
def relu(x): return Activation('relu')(x)
def dropout(x, p): return Dropout(p)(x) if p else x
def bn(x): return BatchNormalization( axis=-1)(x)
def relu_bn(x): return relu(bn(x))
def concat(xs): return concatenate(xs, axis=-1)
def reverse(a): return list(reversed(a))

# convolution+dropout
def conv(x, nf, sz, wd, p, stride=1): 
    x = Conv2D(nf, sz, kernel_initializer='he_uniform', padding='same', 
                      strides=(stride,stride), kernel_regularizer=l2(wd))(x)
    return dropout(x, p)

# bn+relu+convolution+dropout
def conv_relu_bn(x, nf, sz=3, wd=0, p=0, stride=1): 
    return conv(relu_bn(x), nf, sz, wd=wd, p=p, stride=stride)

def dense_block(n,x,growth_rate,p,wd):
    """ the standard dense block
    """
    added = []
    for i in range(n):
        b = conv_relu_bn(x, growth_rate, p=p, wd=wd)
        x = concat([x, b])
        added.append(b)
    return x,added

def transition_dn(x, p, wd):
    """ This is the downsampling transition.
    In the original paper, downsampling consists of 1x1 convolution followed by max pooling. 
    However we've found a stride 2 1x1 convolution to give better results.
    """
#     x = conv_relu_bn(x, x.get_shape().as_list()[-1], sz=1, p=p, wd=wd)
#     return MaxPooling2D(strides=(2, 2))(x)
    return conv_relu_bn(x, x.get_shape().as_list()[-1], sz=1, p=p, wd=wd, stride=2)

def down_path(x, nb_layers, growth_rate, p, wd):
    """
    Next we build the entire downward path, keeping track of Dense block outputs 
    in a list called skip.
    """
    skips = []
    for i,n in enumerate(nb_layers):
        x,added = dense_block(n,x,growth_rate,p,wd)
        skips.append(x)
        x = transition_dn(x, p=p, wd=wd)
    return skips, added

def transition_up(added, wd=0):
    """
    This is the upsampling transition. We use a transpose convolution layer.
    """
    x = concat(added)
    _,r,c,ch = x.get_shape().as_list()
    return Conv2DTranspose(ch, 3,  kernel_initializer='he_uniform', 
               padding='same', strides=(2,2), kernel_regularizer=l2(wd))(x)
#     x = UpSampling2D()(x)
#     return conv(x, ch, 2, wd, 0)

def up_path(added, skips, nb_layers, growth_rate, p, wd):
    """
    This builds our upward path, concatenating the skip connections 
    from skip to the Dense block inputs.
    """
    for i,n in enumerate(nb_layers):
        x = transition_up(added, wd)
        x = concat([x,skips[i]])
        x,added = dense_block(n,x,growth_rate,p,wd)
    return x

#
class Tiramisu(myNet):
    """ tiramisu net
    """
    def __init__(self, img_rows = 512, img_cols = 512, img_nchs = 1, nclasses=1, \
		         out_dir='./results', model_dir='./model', \
                 nb_dense_block=6, \
                 growth_rate=12, nb_filter=48, nb_layers_per_block=4, \
                 p=None, wd=0, **kwargs):
        """
        Arguments:
            nclasses: number of classes
            img_input: tuple of shape (channels, rows, columns) or (rows, columns, channels)
            nb_dense_block: number of dense blocks to add
            growth_rate: number of filters to add per dense block
            nb_filter: initial number of filters
            If positive integer, a set number of layers per dense block.
            If list, nb_layer is used as provided
            p: dropout rate
            wd: weight decay
        """
        super(Tiramisu, self).__init__(img_rows, img_cols, img_nchs, nclasses, \
		                          out_dir, model_dir, **kwargs)

        # special parameters for tiramisu net
        self.nb_dense_block=nb_dense_block
        self.growth_rate=growth_rate
        self.nb_filter=nb_filter
        self.nb_layers_per_block=nb_layers_per_block
        self.p=p
        self.wd=wd

    def get_net(self):

        print("using net {}".format(__name__))

        #
        input_shape = (self.img_rows, self.img_cols, self.img_nchs)
        nclasses = self.nclasses

        #
        img_input = Input(shape=input_shape)

        # for convenience
        nb_dense_block=self.nb_dense_block
        growth_rate=self.growth_rate
        nb_filter=self.nb_filter
        nb_layers_per_block=self.nb_layers_per_block
        p=self.p
        wd=self.wd

    #
        if type(nb_layers_per_block) is list or type(nb_layers_per_block) is tuple:
            nb_layers = list(nb_layers_per_block)
        else: nb_layers = [nb_layers_per_block] * nb_dense_block

        x = conv(img_input, nb_filter, 3, wd, 0)
        skips,added = down_path(x, nb_layers, growth_rate, p, wd)
        x = up_path(added, reverse(skips[:-1]), reverse(nb_layers[:-1]), growth_rate, p, wd)
        
        x = conv(x, nclasses, 1, wd, 0)
        if nclasses == 1:
            x = Activation('sigmoid')(x)
        else:
            x = Reshape((-1, nclasses))(x)
            x = Activation('softmax')(x)

    #
        model = Model(inputs=img_input, outputs=x)

        return model
