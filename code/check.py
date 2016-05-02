import pickle
import sys
import traceback

import codewords

print 'Enter a valid path to a result directory: ',
results_dir = raw_input()

with open(results_dir + '/' + 'kmeans.pkl', 'rb') as f:
    kmeans = pickle.load(f)

with open(results_dir + '/' + 'svm.pkl', 'rb') as f:
    clf = pickle.load(f)

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
