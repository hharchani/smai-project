import pickle
import os
import random
import time

import config

results_dir = '../results/results' + str(int(time.time()))
os.mkdir(results_dir)


def get_file_list():
    categories = os.listdir(config.datasetpath)
    file_list = []
    print 'Building list of images...',
    for category in categories:
        category_basepath = config.datasetpath + category + '/'
        img_name_list = os.listdir(category_basepath)
        img_name_list = map(lambda x: category_basepath + x, img_name_list)
        file_list.extend(img_name_list)
    print 'Done. {} images in dataset.'.format(len(file_list))
    print 'Shuffling list...',
    random.shuffle(file_list)
    print 'Done.'
    print 'Dividing images into test and training data... ',
    num_of_training_sample = int(config.trainpercent * len(file_list) / 100)
    training_files = file_list[0:num_of_training_sample]
    test_files = file_list[num_of_training_sample:]
    print 'Done.'
    print '{} images in training set'.format(len(training_files))
    print '{} images in testing set'.format(len(test_files))
    return training_files, test_files, categories


def dump(what, data):
    print 'Saving {} to disk...'.format(what),
    with open(results_dir + '/' + what + '.pkl', 'wb') as f:
        pickle.dump(data, f)
    print 'Done'
