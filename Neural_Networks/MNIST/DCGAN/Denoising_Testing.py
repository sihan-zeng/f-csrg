'''
Created on Oct 23, 2018

@author: kyle
'''


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # Testing set up
    mnist = tf.keras.datasets.mnist
    (_, _), (test_set, _) = mnist.load_data()
    test_set = (test_set / 255.0)
    np.random.shuffle(test_set)
    num_pics = 10
    canvas_orig = np.empty((28 * num_pics, 28 * num_pics))
    canvas_recon = np.empty((28 * num_pics, 28 * num_pics))
    canvas_gen = np.empty((28 * num_pics, 28 * num_pics))
    
    # Start testing
    with tf.Session() as sess:
        # Restore variables from disk
        loc = "./trained_model_dn/epoch-199"
        saver = tf.train.import_meta_graph(loc + '.meta', clear_devices=True)
        saver.restore(sess, loc)
        graph = tf.get_default_graph()
        Xnoisy = graph.get_tensor_by_name('Xnoisy:0')
        Xdn = graph.get_tensor_by_name('Xdn:0')
        isTrain_DCGAN = graph.get_tensor_by_name('isTrain_Gen:0')
        isTrain_proj = graph.get_tensor_by_name('isTrain_Proj:0')
        print("GAN Model restored.")
        
        for i in range(num_pics):
            # MNIST test set
            batch_x = test_set[i*num_pics:(i+1)*num_pics]
            batch_x = (np.reshape(batch_x, (num_pics, 28, 28, 1))-.5)*2
            batch_xn = batch_x + np.random.randn(num_pics, 28, 28, 1) * 1.5
            xdn = sess.run(Xdn, feed_dict={Xnoisy: batch_xn, isTrain_DCGAN: False, isTrain_proj: False})
            xdn = xdn*.5+.5
    
            # Display original images
            for j in range(num_pics):
                # Draw the original digits
                canvas_orig[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                    batch_xn[j].reshape([28, 28])
            # Display reconstructed images
            for j in range(num_pics):
                # Draw the reconstructed digits
                canvas_recon[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                    xdn[j].reshape([28, 28])
    
        print("Original Images")
        plt.figure(figsize=(num_pics, num_pics))
        plt.imshow(canvas_orig, origin="upper", cmap="gray")
        plt.axis('off')
    
        print("Reconstructed Images")
        plt.figure(figsize=(num_pics, num_pics))
        plt.imshow(canvas_recon, origin="upper", cmap="gray")
        plt.axis('off')
        
    plt.show()