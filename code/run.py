import sys
import traceback

import dataset
import kmeans
import codewords
import svm
import config

print """
###############################################################################
#                                                                             #
# Started file listing                                                        #
#                                                                             #
###############################################################################
"""
file_list, test_file_list = dataset.get_file_list()
if not config.simulate:
    dataset.dump('training_file_list', file_list)
    dataset.dump('testing_file_list', test_file_list)

print """
###############################################################################
#                                                                             #
# File listing done                                                           #
#                                                                             #
# Started kmeans                                                              #
#                                                                             #
###############################################################################
"""
kmeans = kmeans.compute_kmeans(file_list)
if not config.simulate:
    dataset.dump('kmeans', kmeans)
print """
###############################################################################
#                                                                             #
# Kmeans clustering done                                                      #
#                                                                             #
# Started generating codewords                                                #
#                                                                             #
###############################################################################
"""
list_of_codewords = codewords.compute_codewords(file_list, kmeans)
if not config.simulate:
    dataset.dump('codewords', list_of_codewords)

print """
###############################################################################
#                                                                             #
# Codeword generation done                                                    #
#                                                                             #
# Started training svm                                                        #
#                                                                             #
###############################################################################
"""
clf = svm.train_svm(file_list, list_of_codewords)
if not config.simulate:
    dataset.dump('svm', clf)
print """
###############################################################################
#                                                                             #
# Training SVM done                                                           #
#                                                                             #
# Performing Validations                                                      #
#                                                                             #
###############################################################################
"""
num_of_sample_tested = 0
num_of_sample_classified_correctly = 0
for f in test_file_list:
    codeword = codewords.get_codeword(f, kmeans)
    predicted_class = clf.predict([codeword])[0]
    actual_class = f.split('/')[-2]
    num_of_sample_tested += 1
    if predicted_class == actual_class:
        num_of_sample_classified_correctly += 1
    print '[{:5}/{:5}, acc: {:.3f}] Classifying {}: {} ({})'.format(
        num_of_sample_tested,
        len(test_file_list),
        num_of_sample_classified_correctly * 100.0 / num_of_sample_tested,
        f.replace(config.datasetpath, ''),
        predicted_class,
        predicted_class == actual_class
    )
print '\nDone Verifying test samples'

while True:
    print 'Enter any image path to check its class: ',
    try:
        img_path = raw_input()
        if img_path.strip() == '':
            continue
        codeword = codewords.get_codeword(img_path, kmeans)
        predicted_class = clf.predict([codeword])[0]
        print 'Predicted class: {}'.format(predicted_class)
    except (EOFError, KeyboardInterrupt):
        print '\nBye'
        sys.exit(0)
    except:
        print traceback.format_exc()
