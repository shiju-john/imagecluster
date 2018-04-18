import os, re
import numpy as np
import imagecluster
import common

pathJoin = os.path.join


def main(imagedir,processingDir, similarity =.4):

    imageFeaturePath = pathJoin( processingDir, 'imagefeatures.pk')
    if not os.path.exists(imageFeaturePath):
        common.makeDir(imageFeaturePath)
        print("No imagefeatures database {} found".format(imageFeaturePath))
        files = common.get_files(imagedir)
        model = imagecluster.get_model()
        fps = imagecluster.fingerprints(files, model, size=(224, 224))
        common.write_pk(fps, imageFeaturePath)
    else:
        print("loading fingerprints database {} ...".format(imageFeaturePath))
        fps = common.read_pk(imageFeaturePath)
    print("clustering ...")
    imagecluster.make_links(imagecluster.cluster(fps, similarity), pathJoin(imagedir, processingDir, 'clusters'))


main("../../data/innercluster",'../../data/innercluster/imagecluster')