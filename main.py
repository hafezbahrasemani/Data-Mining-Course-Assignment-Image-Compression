import load_images
import k_means

if __name__ == '__main__':
    path1 = 'image.png'
    path2 = 'compressed_image.png'
    path3 = 'compressed_256_image.png'

    # uncomment below lines for all the parts

    # Part A and B
    # load_images.load_and_show_image(path1)
    # load_images.run_k_means(path1, path2, 16, 2)

    # Part C
    load_images.run_k_means(path1, path3, 256, 2)

    # Part D
    load_images.show_images_alongside(path1, path3)
