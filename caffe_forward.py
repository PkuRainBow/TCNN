import numpy as np
import caffe
import matplotlib.pyplot as plt

def vis_square(data):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""

    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
               (0, 1), (0, 1))                 # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    plt.imshow(data);
    plt.show()
    plt.axis('off')


# VGG Model test
model_weights = 'D:\\users\\v-yuhyua\\fromGPU02\\test_model\\VGG_2_pass_iter_2000.caffemodel.h5'
model_def = 'D:\\users\\v-yuhyua\\fromGPU02\\feature\\cifar_VGG_F_extract_fc8.prototxt'

# lenet Model test
# model_weights = 'D:\\users\\v-yuhyua\\fromGPU02\\test_model\\cifar10_full_iter_70000.caffemodel'
# model_def = 'D:\\users\\v-yuhyua\\fromGPU02\\feature\\cifar10_extract_features.prototxt'

net = caffe.Net(model_def, model_weights, caffe.TEST)
caffe.set_mode_gpu()

for layer_name, blob in net.blobs.iteritems():
    print layer_name + '\t' + str(blob.data.shape)

for layer_name, param in net.params.iteritems():
    print layer_name + '\t' + str(param[0].data.shape), str(param[1].data.shape)

net.forward()
#visualize the conv1
# filters = net.params['conv1'][0].data
# vis_square(filters.transpose(0, 2, 3, 1))
#visualize the conv2
# filters = net.params['conv2'][0].data
# vis_square(filters[:64].reshape(64**2, 5, 5))
# feat = net.blobs['conv3'].data[0]
# vis_square(feat, padval=0.5)
# feat = net.blobs['pool5'].data[0]
# vis_square(feat)

# feat = net.blobs['fc6'].data[0]
# plt.subplot(2, 1, 1)
# plt.plot(feat.flat)
# plt.subplot(2, 1, 2)
# _ = plt.hist(feat.flat[feat.flat > 0], bins=100)
# plt.show()

feat = net.blobs['prob'].data[99]
plt.figure(figsize=(15, 3))
plt.plot(feat.flat)
plt.show()

feat = net.blobs['prob'].data[1]
plt.figure(figsize=(15, 3))
plt.plot(feat.flat)
plt.show()

# prob = net.blobs['ip2'].data
# print prob
# im = np.transpose(im, (2,1,0))
# plt.imshow(im)
# plt.show()
# print out