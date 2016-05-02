from sklearn.cluster import MiniBatchKMeans
import config
import sift


def compute_kmeans(file_list):
    kmeans = MiniBatchKMeans(
        n_clusters=config.clustersize,
        batch_size=config.batchsize
    )

    imgs_done = 0
    print 'Computing k-means: {:5}/{:5} ({:.3f}%)'.format(
        imgs_done,
        len(file_list),
        imgs_done * 100.0 / len(file_list)
    ),

    descs_buffer = []
    total_descs = 0

    for img_path in file_list:
        descs = sift.get_descriptors(img_path)
        imgs_done += 1
        if descs is None:
            print 'Oopsie?', img_path, 'No descriptors returned!'
            continue
        if len(descs_buffer) > 0 or len(descs) < config.batchsize:
            descs_buffer.extend(descs)
        else:
            kmeans.partial_fit(descs)
            total_descs += len(descs)

        if len(descs_buffer) >= config.batchsize:
            kmeans.partial_fit(descs_buffer)
            total_descs += len(descs_buffer)
            descs_buffer = []

        print '\rComputing k-means: {:5}/{:5} ({:.3f}%)'.format(
            imgs_done,
            len(file_list),
            imgs_done * 100.0 / len(file_list)
        ),

    print '\nKmeans clustering complete, analysed {} samples'.format(total_descs)
    return kmeans
