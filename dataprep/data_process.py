# preprocess data
import numpy as np
from scipy import ndimage

class DataProcess:
    """ some functions for preprocessing data

    """
    @staticmethod
    def preprocess(imgs, new_h=None, new_w=None, flag_simple=True, flag_mask=False):
        """ preprocess the data, a combination 
        """
        print(' preprocess data:')
        # resize it to new dimension
        if new_h is not None or new_w is not None:   # if either one has a new value
            imgs = DataProcess.resize(imgs, new_h, new_w)

		# normalize it
        imgs = DataProcess.scale(imgs, flag_simple, flag_mask)

        return imgs

    @staticmethod
    def resize(imgs, new_h, new_w, order=3):
        """ resize data
        """
        assert len(imgs.shape) == 4, "has to be in the format of (n, h, w, c)"
        #
        n, h, w, c = imgs.shape

        # need resize?
        if h == new_h and w == new_w:
            print(" No need to resize: h={}, w={}".format(h, w))
            return imgs

        # scale for new size
        if new_h is not None or new_w is not None:
            if new_h == None:
                new_h = h
            if new_w == None:
                new_w = w
        else:
            return imgs

        #
        scale_h = new_h/h
        scale_w = new_w/w

        #
        imgs_p = np.ndarray((n, new_h, new_w, c), dtype=np.uint8)

        # resize it
        for i in range(n):
            imgs_p[i] = ndimage.zoom(imgs[i], (scale_h, scale_w, 1), order=order)

        return imgs_p

    @staticmethod
    def scale(imgs, flag_simple, flag_mask):
        """ balance images
        """
        imgs_p = imgs.astype('float32')
        print('shape=', imgs_p.shape)

        #
        if flag_simple:
            imgs_p /= 255
        else:
            mean = np.mean(imgs_p)
            std  = np.std(imgs_p)
            imgs_p -= mean
            imgs_p /= std	

        if flag_mask:
            imgs_p[imgs_p > 0.5]  = 1
            imgs_p[imgs_p <= 0.5] = 0

        return imgs_p
