'''
Created on Oct 28, 2018

@author: kyle
'''


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class ImportDENGEN(object):
    def __init__(self, loc):
        self.graph = tf.Graph()
        self.sess = tf.Session(graph = self.graph)
        with self.graph.as_default():
            saver = tf.train.import_meta_graph(loc + '.meta', clear_devices=True)
            saver.restore(self.sess, loc)
            self.Xnoisy = self.graph.get_tensor_by_name('Xnoisy:0')
            self.Xdn = self.graph.get_tensor_by_name('Xdn:0')
            self.Xdn = tf.reshape(self.Xdn, [-1, 28*28])         
            self.Z = self.graph.get_tensor_by_name('Proj:0')
            self.isTrain_Gen = self.graph.get_tensor_by_name('isTrain_Gen:0')
            self.isTrain_Proj = self.graph.get_tensor_by_name('isTrain_Proj:0')
            print("DENGEN Model restored.")
    
    def getXRec(self, x_):
        '''
        dim_in: valid variations of [1,28,28,1]
        dim_out: [1, 784]
        '''
        return self.sess.run(self.Xdn, feed_dict={self.Xnoisy: x_.reshape([-1,28,28,1]),
                                                  self.isTrain_Gen: False,
                                                  self.isTrain_Proj: False})
    
    def getZ(self, x_):
        '''
        dim_in: valid variations of [1,28,28,1]
        dim_out: [1, 1, 1, 100]
        '''
        return self.sess.run(self.Z, feed_dict={self.Xnoisy: x_.reshape([-1,28,28,1]),
                                                self.isTrain_Gen: False,
                                                self.isTrain_Proj: False})
        
    def __call__(self, x_):
        '''
        dim_in: valid variations of [1,28,28,1]
        dim_out: [[1, 1, 1, 500], [1, 784]]
        '''
        return self.sess.run([self.Z, self.Xdn], feed_dict={self.Xnoisy: x_.reshape([-1,28,28,1]),
                                                            self.isTrain_Gen: False,
                                                            self.isTrain_Proj: False})


if __name__ == '__main__':
    # Testing set up
    mnist = tf.keras.datasets.mnist
    (_, _), (test_set, _) = mnist.load_data()
    test_set = (test_set / 255.0)
    np.random.shuffle(test_set)
    num_pics = 1
    canvas_orig = np.empty((28 * num_pics, 28 * num_pics))
    canvas_recon = np.empty((28 * num_pics, 28 * num_pics))
    canvas_gen = np.empty((28 * num_pics, 28 * num_pics))
    
    # Start testing
    loc = './trained_model_dn/epoch-499'
    DENGEN = ImportDENGEN(loc)
    for i in range(num_pics):
        # MNIST test set
        batch_x = test_set[i*num_pics:(i+1)*num_pics]
        batch_x = (np.reshape(batch_x, (num_pics, 28, 28, 1))-.5)*2
        batch_xn = batch_x + np.random.randn(num_pics, 28, 28, 1) * .5
        xdn = DENGEN.getXRec(batch_xn)
        z, xdn = DENGEN(batch_xn)
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
    
    plt.figure()
    plt.imshow(canvas_orig, origin="upper", cmap="gray")
    plt.axis('off')
    plt.title("Noisy Images")
    
    plt.figure()
    plt.imshow(canvas_recon, origin="upper", cmap="gray")
    plt.axis('off')
    plt.title("Denoised Images")
            
    plt.show()