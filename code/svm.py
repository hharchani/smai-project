from sklearn import svm


def train_svm(file_list, codewords):
    clf = svm.SVC()
    clf.fit(codewords, [file_path.split('/')[-2] for file_path in file_list])
    return clf
