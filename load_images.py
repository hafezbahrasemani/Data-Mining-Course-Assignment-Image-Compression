import matplotlib.pyplot as plt
from matplotlib import image
import numpy as np
import k_means
import os


def load_and_show_image(path):
    img = image.imread(path)
    plt.imshow(img)
    plt.show()

    print("Shape of the image is: " + str(img.shape))


def show_images_alongside(path1, path2):
    img1 = image.imread(path1)
    img2 = image.imread(path2)

    size1 = os.path.getsize(path1)
    size2 = os.path.getsize(path2)

    f = plt.figure()
    f.add_subplot(1, 2, 1)
    plt.imshow(img1)
    f.add_subplot(1, 2, 2)
    plt.imshow(img2)
    plt.suptitle("Original_Size: " + str(size1)
                 + "    Compressed_Size: " + str(size2) + "     (in bytes)")
    plt.show(block=True)


def run_k_means(path1, path2, k, max_iteration):
    # img_2d = np.zeros()
    img = image.imread(path1)

    img_2d = np.reshape(img, ((img.shape[0] * img.shape[1]), 3))

    clusters, centers, data_from_cluster = \
        k_means.specifying_clusters(img_2d, k, max_iteration)

    print(centers)
    # print(data_from_cluster)

    # part B
    for i in range(len(img_2d)):
        img_2d[i] = centers[int(data_from_cluster[i])]

    # save compressed image
    img_2d_p = np.reshape(img_2d, img.shape)
    image.imsave(path2, img_2d_p)
