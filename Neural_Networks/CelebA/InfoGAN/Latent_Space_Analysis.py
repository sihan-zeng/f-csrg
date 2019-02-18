'''
Created on Oct 23, 2018

@author: Sihan & kyle
'''


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


MODEL_LOC = 'trained_model_dn_infogan/epoch-212'


def createFigure(fs=(8, 6)):
    plt.figure(figsize=fs)
    plt.rc('font', size=16)          # controls default text sizes
    plt.rc('axes', titlesize=16)     # fontsize of the axes title
    plt.rc('axes', labelsize=18)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=16)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=16)    # fontsize of the tick labels
    plt.rc('legend', fontsize=12)    # legend fontsize
    plt.rc('figure', titlesize=18)  # fontsize of the figure title
    
    
def sample_disc_c(n_samples, num_codes, cat_size):
    # returns n_samples by (num_codes*cat_size) array
    return np.random.multinomial(1, cat_size*[1./cat_size], size=n_samples*num_codes).reshape((n_samples,-1))


class ImportGEN():
    def __init__(self, loc):
        self.graph = tf.Graph()
        self.sess = tf.Session(graph = self.graph)
        with self.graph.as_default():
            saver = tf.train.import_meta_graph(loc + '.meta', clear_devices=True)
            saver.restore(self.sess, loc)
            self.isTrain_Gen = self.graph.get_tensor_by_name('isTrain_Gen:0')
            self.Z = self.graph.get_tensor_by_name('Proj:0')
            self.Xdn = self.graph.get_tensor_by_name('Xdn:0')
            print("GAN Model restored.")
    
    def getXGen(self, z_):
        return self.sess.run(self.Xdn, feed_dict={self.Z: z_, self.isTrain_Gen: False})


if __name__ == '__main__':
    # Testing set up
    num_pics_per_col = 10
    num_pics_per_row = 25
    canvas_gen = np.empty((32 * num_pics_per_col, 32 * num_pics_per_row))

    DENGEN = ImportGEN(MODEL_LOC)
        
    for i in range(num_pics_per_col):        
        # fix codeword and vary only noise
        c_cont = np.random.randn(1,5)
        c_cat = sample_disc_c(1,5,10)
        c = np.hstack((np.tile(c_cont,(num_pics_per_row,1)),np.tile(c_cat,(num_pics_per_row,1))))
        z_noise = np.random.randn(num_pics_per_row, 128)
        z = np.hstack((c,z_noise))
        z = z.reshape([-1, 1, 1, 183])
        
        xg = DENGEN.getXGen(z)

        # Display generated images
        for j in range(num_pics_per_row):
            # Draw the reconstructed digits
            canvas_gen[i * 32:(i + 1) * 32, j * 32:(j + 1) * 32] = \
                xg[j].reshape([32, 32])
    
    createFigure(fs=(num_pics_per_row/2, num_pics_per_col/2))
    plt.imshow(canvas_gen, origin="upper", cmap="gray")
    plt.axis('off')
    plt.title("Fix codeword and vary only noise")
    
    for i in range(num_pics_per_col):        
        # fix noise and vary only codeword
        z_noise = np.random.randn(1, 128)
        z_noise = np.tile(z_noise,(num_pics_per_row, 1))
        c_cont = np.random.randn(num_pics_per_row, 5)
        c_cat = sample_disc_c(num_pics_per_row,5, 10)
        c = np.hstack((c_cont, c_cat))
        z = np.hstack((c, z_noise))
        z = z.reshape([-1, 1, 1, 183])
        
        xg = DENGEN.getXGen(z)

        # Display generated images
        for j in range(num_pics_per_row):
            # Draw the reconstructed digits
            canvas_gen[i * 32:(i + 1) * 32, j * 32:(j + 1) * 32] = \
                xg[j].reshape([32, 32])
    
    createFigure(fs=(num_pics_per_row/2, num_pics_per_col/2))
    plt.imshow(canvas_gen, origin="upper", cmap="gray")
    plt.axis('off')
    plt.title("Fix noise and vary only codeword")
    
    
    plt.show()
        
    