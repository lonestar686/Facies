from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np 
import os
import glob
import cv2
#from libtiff import TIFF

# the path for original data
DATA_PATH = './data'

# the path for temporary augmented data
DEFORM_PATH = './deform'

# the path for training data that will be used by nn
NPY_PATH  = os.path.join(DEFORM_PATH, "npydata")

class myAugmentation(object):
	
	"""
	A class used to augmentate image
	Firstly, read train image and label seperately, and then merge them together for the next process
	Secondly, use keras preprocessing to augmentate image
	Finally, seperate augmentated image apart into train image and label
	"""

	def __init__(self, data_path=DATA_PATH, \
					   image_path="train/image", label_path="train/label", \
					   deform_path=DEFORM_PATH, \
		               merge_path="merge", aug_merge_path="aug_merge", \
					   aug_image_path="train/image", aug_label_path="train/label", \
					   img_type="tif"):
		
		"""
		Using glob to get all .img_type form path
		"""
		#
		self.img_type = img_type
		#
		self.image_path = os.path.join(data_path, image_path)
		self.label_path = os.path.join(data_path, label_path)

		# files for original images
		self.train_imgs = glob.glob(self.image_path+"/*."+img_type)
		self.label_imgs = glob.glob(self.label_path+"/*."+img_type)
		#
		self.slices = len(self.train_imgs)

		# temporary paths
		self.merge_path     = os.path.join(deform_path, merge_path)
		if not os.path.exists(self.merge_path):
			os.makedirs(self.merge_path)
		#
		self.aug_merge_path = os.path.join(deform_path, aug_merge_path)
		if not os.path.exists(self.aug_merge_path):
			os.makedirs(self.aug_merge_path)

		# augmented data paths
		self.aug_image_path = os.path.join(deform_path, aug_image_path)
		if not os.path.exists(self.aug_image_path):
			os.makedirs(self.aug_image_path)
		#
		self.aug_label_path = os.path.join(deform_path, aug_label_path)
		if not os.path.exists(self.aug_label_path):
			os.makedirs(self.aug_label_path)

		#
		self.datagen = ImageDataGenerator(
							        rotation_range=0.2,
							        width_shift_range=0.05,
							        height_shift_range=0.05,
							        shear_range=0.05,
							        zoom_range=0.05,
							        horizontal_flip=True,
							        fill_mode='nearest')

	def Augmentation(self):
		"""
		Start augmentation.....
		"""
		#
		imgtype = self.img_type
		#
		images = self.train_imgs
		labels = self.label_imgs

		# temporary paths
		path_merge     = self.merge_path
		path_aug_merge = self.aug_merge_path

		if len(images) != len(labels) or len(images) == 0 or len(labels) == 0:
			print("# of train images don't match their labels")
			return 0

		for i in range(len(images)):
			#
			img_t = load_img(images[i])    # Note: load it with grayscale=False
			img_l = load_img(labels[i])

			#
			x_t = img_to_array(img_t)
			x_l = img_to_array(img_l)
			x_t[:,:,2] = x_l[:,:,0]
			img_tmp = array_to_img(x_t)

			# save merged image
			img_tmp.save(path_merge+"/"+str(i)+"."+imgtype)

			# augment data
			img = x_t
			img = img.reshape((1,) + img.shape)

			#
			savedir = path_aug_merge + "/" + str(i)
			if not os.path.lexists(savedir):
				os.mkdir(savedir)

			self.doAugmentate(img, savedir, str(i))


	def doAugmentate(self, img, save_to_dir, save_prefix, batch_size=1, save_format='tif', imgnum=30):

		"""
		augmentate one image
		"""
		datagen = self.datagen
		i = 0
		for batch in datagen.flow(img,
									batch_size=batch_size,
									save_to_dir=save_to_dir,
									save_prefix=save_prefix,
									save_format=save_format):
		    i += 1
		    if i > imgnum:
		        break

	def splitMerge(self):

		"""
		split merged image apart
		"""
		aug_merge_path = self.aug_merge_path

		# splitted images/labels
		aug_image_path = self.aug_image_path
		aug_label_path = self.aug_label_path

		#
		for i in range(self.slices):

			# current merged data
			path = aug_merge_path + "/" + str(i)
			train_imgs = glob.glob(path+"/*."+self.img_type)

			# augemented images
			savedir_image = aug_image_path #+ "/" + str(i)
			if not os.path.lexists(savedir_image):
				os.mkdir(savedir_image)
			# augmented labels
			savedir_label = aug_label_path #+ "/" + str(i)
			if not os.path.lexists(savedir_label):
				os.mkdir(savedir_label)
			#
			for imgname in train_imgs:
				#
				img = cv2.imread(imgname)
				img_train = img[:,:,2]#cv2 read image rgb->bgr
				img_label = img[:,:,0]
				#
				midname = imgname[imgname.rindex("/")+1:imgname.rindex("."+self.img_type)]
				cv2.imwrite(savedir_image+"/"+midname+"_aug"+"."+self.img_type,img_train)
				cv2.imwrite(savedir_label+"/"+midname+"_aug"+"."+self.img_type,img_label)

	def splitTransform(self):

		"""
		split perspective transform images
		"""
		#path_merge = "transform"
		#path_train = "transform/data/"
		#path_label = "transform/label/"
		path_merge       = "deform/deform_norm2"
		path_train_image = "deform/train/image/"
		path_train_label = "deform/train/label/"

		#
		train_imgs = glob.glob(path_merge+"/*."+self.img_type)
		#
		for imgname in train_imgs:

			img = cv2.imread(imgname)
			img_train = img[:,:,2]#cv2 read image rgb->bgr
			img_label = img[:,:,0]
			#
			midname = imgname[imgname.rindex("/")+1:imgname.rindex("."+self.img_type)]
			cv2.imwrite(path_train_image+midname+"."+self.img_type,img_train)
			cv2.imwrite(path_train_label+midname+"."+self.img_type,img_label)

# get image dimensions
def get_height_width_tuple(img_name):

	# PIL image has format (width, height, channel)
	img = load_img(img_name, grayscale=True)

	return img.size[1], img.size[0]    # height, width

# convert existing training/test data to .npy format
class dataProcess(object):

	def __init__(self, deform_path = DEFORM_PATH,  \
		               image_path = "train/image", \
		               label_path = "train/label", \
					   data_path = DATA_PATH, \
					   test_path = "test",\
					   img_type = "tif", \
					   npy_path = NPY_PATH, \
					   train_npy_name='imgs_train.npy', \
					   label_npy_name='imgs_mask_train.npy',\
					   test_npy_name='imgs_test.npy'):

		"""
		
		"""
		#
		self.image_path = os.path.join(deform_path, image_path)
		self.label_path = os.path.join(deform_path, label_path)
		#
		self.test_path  = os.path.join(data_path, test_path)
		self.img_type   = img_type

		# npy output
		self.npy_path = npy_path
		if not os.path.exists(self.npy_path):
			os.makedirs(self.npy_path)
		#
		self.train_npy_name = os.path.join(npy_path, train_npy_name)
		self.label_npy_name = os.path.join(npy_path, label_npy_name)
		self.test_npy_name  = os.path.join(npy_path, test_npy_name)

	def create_train_data(self):

		print('-'*30)
		print('Creating training images...')
		print('-'*30)

		#
		imgs = glob.glob(self.image_path+"/*."+self.img_type)
		print('total images for training: ', len(imgs))

		# get image dimensions
		out_rows, out_cols = get_height_width_tuple(imgs[0])

		# Note: we assume all of images has the same dimension
		imgdatas = np.ndarray((len(imgs),out_rows,out_cols,1), dtype=np.uint8)
		imglabels = np.ndarray((len(imgs),out_rows,out_cols,1), dtype=np.uint8)
		#
		i = 0
		for imgname in imgs:
			#
			midname = imgname[imgname.rindex("/")+1:]
			img   = load_img(self.image_path + "/" + midname,grayscale = True)
			label = load_img(self.label_path + "/" + midname,grayscale = True)
			img   = img_to_array(img)
			label = img_to_array(label)
			#img = cv2.imread(self.image_path + "/" + midname,cv2.IMREAD_GRAYSCALE)
			#label = cv2.imread(self.label_path + "/" + midname,cv2.IMREAD_GRAYSCALE)
			#img = np.array([img])
			#label = np.array([label])
			imgdatas[i]  = img
			imglabels[i] = label
			if i % 100 == 0:
				print('Done: {0}/{1} images'.format(i, len(imgs)))
			i += 1
		print('loading done')

		np.save(self.train_npy_name, imgdatas)
		np.save(self.label_npy_name, imglabels)
		print('Saving to {} and {} files done.'.format(self.train_npy_name, self.label_npy_name))

	def create_test_data(self):

		print('-'*30)
		print('Creating test images...')
		print('-'*30)
		#
		imgs = glob.glob(self.test_path+"/*."+self.img_type)
		print('total images for testing: ', len(imgs))

		# get image dimensions
		out_rows, out_cols = get_height_width_tuple(imgs[0])

		#
		imgdatas = np.ndarray((len(imgs),out_rows,out_cols,1), dtype=np.uint8)

		i = 0
		for imgname in imgs:
			midname = imgname[imgname.rindex("/")+1:]
			img = load_img(self.test_path + "/" + midname,grayscale = True)
			img = img_to_array(img)
			#img = cv2.imread(self.test_path + "/" + midname,cv2.IMREAD_GRAYSCALE)
			#img = np.array([img])
			imgdatas[i] = img
			i += 1
		print('loading done')
		np.save(self.test_npy_name, imgdatas)
		print('Saving to {} file done.'.format(self.test_npy_name))

	def load_train_data(self):
		print('-'*30)
		print('load train images...')
		print('-'*30)
		imgs_train = np.load(self.train_npy_name)
		imgs_mask_train = np.load(self.label_npy_name)

		# imgs_train = imgs_train.astype('float32')
		# imgs_mask_train = imgs_mask_train.astype('float32')
		# imgs_train /= 255
		# #mean = imgs_train.mean(axis = 0)
		# #imgs_train -= mean	
		# imgs_mask_train /= 255
		# imgs_mask_train[imgs_mask_train > 0.5] = 1
		# imgs_mask_train[imgs_mask_train <= 0.5] = 0

		return imgs_train,imgs_mask_train

	def load_test_data(self):
		print('-'*30)
		print('load test images...')
		print('-'*30)
		imgs_test = np.load(self.test_npy_name)

		# imgs_test = imgs_test.astype('float32')
		# imgs_test /= 255
		# #mean = imgs_test.mean(axis = 0)
		# #imgs_test -= mean	

		return imgs_test

if __name__ == "__main__":

	aug = myAugmentation()
	aug.Augmentation()
	aug.splitMerge()
	#aug.splitTransform()
	mydata = dataProcess()
	mydata.create_train_data()
	mydata.create_test_data()
	#imgs_train,imgs_mask_train = mydata.load_train_data()
	#print imgs_train.shape,imgs_mask_train.shape
