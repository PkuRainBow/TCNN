import caffe
import lmdb
import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
from caffe.proto import caffe_pb2

# get the index of the max probabily output
def binarize(v):
    one_hot = np.zeros(10)
    one_hot[np.argmax(v)] = 1
    return one_hot

def get_precisions(data_source):
    [data_vector, train_vector, test_vector] = pickle.load(open('resize_label.dat', 'r'))
    if data_source == "data" :
        num = 59000
        feature_lmdb_env = lmdb.open('D:\\users\\v-yuhyua\\fromGPU02\\feature\\SGD_iter20000_data_lmdb')
        label_vector = data_vector
    elif data_source == "test" :
        num = 1000
        feature_lmdb_env = lmdb.open('D:\\users\\v-yuhyua\\fromGPU02\\feature\\SGD_iter20000_test_lmdb')
        label_vector = test_vector
    else :
        num = 5000
        feature_lmdb_env = lmdb.open('D:\\users\\v-yuhyua\\fromGPU02\\feature\\SGD_iter20000_train_lmdb')
        label_vector = train_vector

    predict_label = np.zeros(num, dtype=np.int16)

    #get the feature of the validate data
    lmdb_txn = feature_lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    datum = caffe_pb2.Datum()
    for ix,(key, value) in enumerate(lmdb_cursor):
        datum.ParseFromString(value)
        data = caffe.io.datum_to_array(datum)
        data = np.squeeze(data)[:]
        predict_label[ix] = np.argmax(data)
        if (ix+1)%1000==0:
            print 'feature process %d' %(ix+1)

    # get the label of the validate data
    # lmdb_txn = label_lmdb_env.begin()
    # lmdb_cursor = lmdb_txn.cursor()
    # datum = caffe_pb2.Datum()
    # for ix,(key, value) in enumerate(lmdb_cursor):
    #     datum.ParseFromString(value)
    #     data_label_vector[ix] = datum.label
    #     if (ix+1)%1000==0:
    #         print 'label process %d' %(ix+1)

    # Calculate the precision
    count = 0
    for i in xrange(num):
        if  predict_label[i] == label_vector[i]:
            count = count + 1
    precision = count / float(num)
    print 'data precision %f' % (precision)

def store_label():
    data_label_lmdb_env = lmdb.open('D:\\users\\v-yuhyua\\fromGPU02\\lmdb\\resize_data_cifar10_lmdb')
    test_label_lmdb_env = lmdb.open('D:\\users\\v-yuhyua\\fromGPU02\\lmdb\\resize_test_cifar10_lmdb')
    train_label_lmdb_env = lmdb.open('D:\\users\\v-yuhyua\\fromGPU02\\lmdb\\resize_train_cifar10_lmdb')

    data_vector = np.zeros(59000, dtype=np.int16)
    train_vector = np.zeros(5000, dtype=np.int16)
    test_vector = np.zeros(1000, dtype=np.int16)

    lmdb_txn = data_label_lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    datum = caffe_pb2.Datum()
    for ix, (key, value) in enumerate(lmdb_cursor):
        datum.ParseFromString(value)
        data_vector[ix] = datum.label
        if (ix + 1) % 1000 == 0:
            print 'data label process %d' % (ix + 1)


    lmdb_txn = train_label_lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    datum = caffe_pb2.Datum()
    for ix, (key, value) in enumerate(lmdb_cursor):
        datum.ParseFromString(value)
        train_vector[ix] = datum.label
        if (ix + 1) % 1000 == 0:
            print 'train label process %d' % (ix + 1)

    lmdb_txn = test_label_lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    datum = caffe_pb2.Datum()
    for ix, (key, value) in enumerate(lmdb_cursor):
        datum.ParseFromString(value)
        test_vector[ix] = datum.label
        if (ix + 1) % 1000 == 0:
            print 'label process %d' % (ix + 1)
    pickle.dump([data_vector, train_vector, test_vector], open('resize_label.dat', 'wb'), True)

get_precisions("data")
# get_precisions("train")
#store_label()
