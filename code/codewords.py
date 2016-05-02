import numpy as np
import config
import sift


def get_codeword(img_path, kmeans):
    codeword = np.zeros(config.clustersize, dtype='int32')
    descs = sift.get_descriptors(img_path)
    if descs is not None:
        for desc in descs:
            n = kmeans.predict([desc])
            codeword[n] += 1
    return codeword


def compute_codewords(file_list, kmeans):
    imgs_done = 0
    print 'Computing codewords: {:5}/{:5} ({:.3f}%)'.format(
        imgs_done,
        len(file_list),
        imgs_done * 100.0 / len(file_list)
    ),

    codewords = []

    for img_path in file_list:
        codeword = get_codeword(img_path, kmeans)
        codewords.append(codeword)
        imgs_done += 1

        print '\rComputing codewords: {:5}/{:5} ({:.3f}%)'.format(
            imgs_done,
            len(file_list),
            imgs_done * 100.0 / len(file_list)
        ),

    print '\nComputed {} codewords'.format(len(codewords))
    return codewords
