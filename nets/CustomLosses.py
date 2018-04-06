
from keras import backend as K

smooth = 1.

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def gumble_loss(y_true, y_pred):
    """ 1-DICE + BCE
    """
    dice = dice_loss(y_true, y_pred)
    bce = K.binary_crossentropy(y_true, y_pred)
    return 1+dice + bce
