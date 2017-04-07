
# coding: utf-8

# In[35]:

#get_ipython().magic(u'matplotlib inline')
import caffe
import numpy as np



# set display defaults
#plt.rcParams['figure.figsize'] = (6, 6)        # large images
#plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels


# In[36]:

caffe.set_mode_gpu()


# In[37]:

model_def = "/home/sagar97/Downloads/deep-learning-traffic-lights-master/model/deploy.prototxt"
model_weights = "/home/sagar97/Downloads/deep-learning-traffic-lights-master/model/train_squeezenet_trainval_manual_p2__iter_3817.caffemodel"


# In[38]:

def class_idx_to_name(idx):
    return ['none', 'red', 'green'][idx]


# In[39]:

from caffe.classifier import Classifier


# In[40]:

c = Classifier(
           model_def, 
           model_weights, 
           mean=np.array([104, 117, 123]),
           raw_scale=255,
           channel_swap=(2,1,0),
           image_dims=(256, 256)
)


# In[41]:

# set batch size
BATCH_SIZE = 1
c.blobs['data'].reshape(BATCH_SIZE, 3, c.blobs['data'].shape[2], c.blobs['data'].shape[3])
c.blobs['prob'].reshape(BATCH_SIZE, 3)
c.reshape()


# In[42]:

import os, random

images_path = '/home/sagar97/Downloads/deep-learning-traffic-lights-master/images/'

f = random.choice(os.listdir(images_path))
#print f
image = caffe.io.load_image(images_path + f)
cls = c.predict([image]).argmax()
#plt.imshow(image)
#plt.axis('off')
print 'predicted class is:', class_idx_to_name(cls)
print cls

    
    

     


