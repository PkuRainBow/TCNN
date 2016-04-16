import caffe
import lmdb
import numpy as np
from caffe.proto import caffe_pb2

def normalize_1d(v) :
    norm = np.linalg.norm(v)
    if norm==0:
        return v
    return v/norm

def normalize_2d(v):
    rows,cols=v.shape()
    for i in range(rows):
        v[i,:] = normalize_1d(v[i,:])
    return v

def compuate_distances(query_list, set_list):
    num_query = query_list.shape[0]
    num_set = set_list.shape[0]
    dists = np.zeros((num_query, num_set))
    M = np.dot(query_list, set_list.T)
    te = np.square(query_list).sum(axis = 1)
    tr = np.square(set_list).sum(axis = 1)
    dists = np.sqrt(-2*M + tr + np.matrix(te).T)
    # for i in xrange(num_query):
    #     for j in xrange(num_set):
    #         dists[i,j] = np.linalg.norm(query_list[i,:] - set_list[j,:])

    return dists

def get_precision(dists, test_label_vector, data_label_vector, top_count):
    num_query = dists.shape[0]
    precision = np.zeros((num_query, 1))
    correct_radio = np.zeros((num_query, 1))
    for i in xrange(num_query):
        #print label_vector[i]
        labels_sorted = data_label_vector[np.argsort(dists[i,:])].flatten()
        labels = labels_sorted[0:top_count]
        correct_count = 0
        for j in xrange(top_count):
            #print label[0], label_vector[i][0]
            if labels[j] == test_label_vector[i][0]:
                correct_count = correct_count + 1
                precision[i] += correct_count / float(j+1)
        if correct_count > 0 :
            precision[i] = precision[i] / float(correct_count)
        #print precision[i]
    ave_precision = np.mean(precision)
    return ave_precision

def load_data():
    #feature_lmdb_env = lmdb.open('Z:\\caffe_msra_gpu02\\examples\\cifar10\\triplet_loss\\features_triplet_lmdb')
    #feature_lmdb_env = lmdb.open('Z:\\caffe_msra_gpu02\\examples\\cifar10\\hash_test_cifar10lmdb')
    #feature_lmdb_env = lmdb.open('D:\\users\\v-yuhyua\\fromGPU02\\feature\\triplet_v5_lmdb')

    feature_lmdb_env = lmdb.open('D:\\users\\v-yuhyua\\fromGPU02\\feature\\resize_data_4096_lmdb')
    label_lmdb_env = lmdb.open('D:\\users\\v-yuhyua\\fromGPU02\\lmdb\\resize_data_cifar10_lmdb')
    """
    get the label of the validate data
    """
    lmdb_txn = label_lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    datum = caffe_pb2.Datum()
    num = 59000
    data_label_vector = np.zeros((num,1), dtype=np.int16)
    for ix,(key, value) in enumerate(lmdb_cursor):
        datum.ParseFromString(value)
        data_label_vector[ix, :] = datum.label
        if (ix+1)%1000==0:
            print 'label process %d' %(ix+1)

    """
    get the feature of the validate data
    """
    lmdb_txn = feature_lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    datum = caffe_pb2.Datum()
    num = 59000
    data_feature_vector = np.zeros((num, 4096), dtype=np.float16)
    for ix,(key, value) in enumerate(lmdb_cursor):
        datum.ParseFromString(value)
        data = caffe.io.datum_to_array(datum)
        data = np.squeeze(data)[:]
        data_feature_vector[ix, :] = normalize_1d(data)
        #data_feature_vector[ix, :] = data
        if (ix+1)%1000==0:
            print 'feature process %d' %(ix+1)
    return data_label_vector, data_feature_vector

def load_test():
    feature_lmdb_env = lmdb.open('D:\\users\\v-yuhyua\\fromGPU02\\feature\\resize_test_4096_lmdb')
    label_lmdb_env = lmdb.open('D:\\users\\v-yuhyua\\fromGPU02\\lmdb\\resize_test_cifar10_lmdb')
    """
    get the label of the validate data
    """
    lmdb_txn = label_lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    datum = caffe_pb2.Datum()
    num = 1000
    test_label_vector = np.zeros((num,1), dtype=np.int16)
    for ix,(key, value) in enumerate(lmdb_cursor):
        datum.ParseFromString(value)
        test_label_vector[ix, :] = datum.label
        if (ix+1)%1000==0:
            print 'label process %d' %(ix+1)

    """
    get the feature of the validate data
    """
    lmdb_txn = feature_lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    datum = caffe_pb2.Datum()
    num = 1000
    test_feature_vector = np.zeros((num, 4096), dtype=np.float16)
    for ix,(key, value) in enumerate(lmdb_cursor):
        datum.ParseFromString(value)
        data = caffe.io.datum_to_array(datum)
        data = np.squeeze(data)[:]
        test_feature_vector[ix, :] = normalize_1d(data)
        #data_feature_vector[ix, :] = data
        if (ix+1)%1000==0:
            print 'feature process %d' %(ix+1)
    return test_label_vector, test_feature_vector

""" load the feature and label of the test data and validate data"""
data_label_vector, data_feature_vector = load_data()
test_label_vector, test_feature_vector = load_test()

dists = compuate_distances(test_feature_vector, data_feature_vector)

precision = get_precision(dists, test_label_vector, data_label_vector, 59000)
print precision

