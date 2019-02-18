'''
Created on Dec 26, 2017

@author: kyle
'''


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # Testing set up
    mnist = tf.keras.datasets.mnist
    (_, _), (test_set, lb) = mnist.load_data()
    test_set = np.float32(test_set) / 255.0
    idx_all = np.arange(len(test_set))
    np.random.shuffle(idx_all)
    test_set = test_set[idx_all]
    lb = lb[idx_all]
    num_pics = 1
    canvas_orig = np.empty((28 * num_pics, 28 * num_pics))
    canvas_recon = np.empty((28 * num_pics, 28 * num_pics))
    
    # Start testing
    with tf.Session() as sess:
        
        # Restore variables from disk.
        saver = tf.train.import_meta_graph('trained_model/iter-90000.meta')
        saver.restore(sess, 'trained_model/iter-90000')
        graph = tf.get_default_graph()
        X = graph.get_tensor_by_name('X:0')
        XRec = graph.get_tensor_by_name('XRec:0')
        XDN = tf.nn.sigmoid(graph.get_tensor_by_name('XTrain:0'))
        Noise = graph.get_tensor_by_name('Noise:0')
        NC = graph.get_tensor_by_name('Noise_Coeff:0')
        print("Model restored.")
        
        for i in range(num_pics):
            # MNIST test set
            batch_x = test_set[num_pics*i:num_pics*(i+1)]
            batch_x = batch_x.reshape([num_pics,-1])
            batch_x += np.random.rand(num_pics, 784)*.5
            # Encode and decode the digit image
            loss_AE = tf.reduce_sum(tf.square(XRec - X), axis=1, name='loss_AE')
            g, lae = sess.run([XRec, loss_AE], feed_dict={X: batch_x})
    
            # Display original images
            for j in range(num_pics):
                # Draw the original digits
                canvas_orig[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                    batch_x[j].reshape([28, 28])
            # Display reconstructed images
            for j in range(num_pics):
                # Draw the reconstructed digits
                canvas_recon[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                    g[j].reshape([28, 28])
    
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(canvas_orig, cmap='gray')
        plt.title("Original Images")
        plt.subplot(1,2,2)
        plt.imshow(canvas_recon, cmap='gray')
        plt.title("Reconsturcted Images")
        
        print("L2 Loss: " + str(lae.sum()))
        
        plt.show()