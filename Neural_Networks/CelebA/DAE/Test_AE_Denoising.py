'''
Created on Feb 15, 2019

@author: Sihan
'''


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    x_test = np.load('../../../Datasets/CelebA/Test.npy')

    img_test = x_test[1].reshape((-1,32,32,1))

    with tf.Session() as sess:
        loc = "trained_AE_dn/epoch-499"
        saver = tf.train.import_meta_graph(loc + '.meta', clear_devices=True)
        saver.restore(sess, loc)
        graph = tf.get_default_graph()

        isTrain = graph.get_tensor_by_name('isTrain:0')

        Xnoisy = graph.get_tensor_by_name('Xnoisy:0')
        Xdn = graph.get_tensor_by_name('Xdn:0')

        x_noisy = img_test+np.random.randn(1,32,32,1)*.1

        x_denoised = sess.run(Xdn, feed_dict={Xnoisy: x_noisy, isTrain: False})



        plt.figure()
        plt.subplot(1,3,1)
        plt.imshow(img_test.reshape((32,32)), origin="upper", cmap="gray")
        plt.title('Original')
        plt.subplot(1,3,2)
        plt.imshow(x_noisy.reshape((32,32)), origin="upper", cmap="gray")       
        plt.title('Noisy')
        plt.subplot(1,3,3)
        plt.imshow(x_denoised.reshape((32,32)), origin="upper", cmap="gray")       
        plt.title('Denoised')
        plt.show()


