'''
Created on Oct 23, 2018

@author: kyle
'''


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


MODEL_LOC = './trained_model_infogan/epoch-300'


def createFigure(fs=(8, 6)):
    plt.figure(figsize=fs)
    plt.rc('font', size=16)          # controls default text sizes
    plt.rc('axes', titlesize=16)     # fontsize of the axes title
    plt.rc('axes', labelsize=18)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=16)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=16)    # fontsize of the tick labels
    plt.rc('legend', fontsize=12)    # legend fontsize
    plt.rc('figure', titlesize=18)  # fontsize of the figure title


def sampleLatenVariable(N, cat=None, c1=None, c2=None, rand=None):
    laten_var = np.zeros([N, 1, 1, 74])
    for i in range(N):
        if cat is None:
            code_cat = np.random.randint(10)
        else:
            code_cat = cat
            
        if c1 is None:
            code_c1 = np.random.uniform(-1,1)
        else:
            code_c1 = c1
            
        if c2 is None:
            code_c2 = np.random.uniform(-1,1)
        else:
            code_c2 = c2
            
        if rand is None:
            code_rand = np.random.randn(62)
        else:
            code_rand = rand
            
        laten_var[i, 0, 0, code_cat] = 1.
        laten_var[i, 0, 0, 10] = code_c1
        laten_var[i, 0, 0, 11] = code_c2
        laten_var[i, 0, 0, 12:] = code_rand
        
    return laten_var


class ImportGEN():
    def __init__(self, loc):
        self.graph = tf.Graph()
        self.sess = tf.Session(graph = self.graph)
        with self.graph.as_default():
            saver = tf.train.import_meta_graph(loc + '.meta', clear_devices=True)
            saver.restore(self.sess, loc)
            self.Z = self.graph.get_tensor_by_name('Z:0')
            self.XGen = self.graph.get_tensor_by_name('generator/gen:0')
            self.XGen = tf.reshape(self.XGen, [-1, 28*28])
            self.isTrain = self.graph.get_tensor_by_name('isTrain:0')
            print("GAN Model restored.")
    
    def getXGen(self, z_):
        '''
        dim_in: [1, 1, 1, num_features] 
        dim_out: [1, 784]
        '''
        return self.sess.run(self.XGen, feed_dict={self.Z: z_, self.isTrain: False})


if __name__ == '__main__':
    # Testing set up
    num_pics_per_col = 10
    num_pics_per_row = 25
    canvas_gen = np.empty((28 * num_pics_per_col, 28 * num_pics_per_row))

    DENGEN = ImportGEN(MODEL_LOC)
        
    for i in range(num_pics_per_col):
        z = sampleLatenVariable(num_pics_per_row, cat=i%10, c1=np.random.uniform(-1,1), c2=np.random.uniform(-1,1))
        xg = DENGEN.getXGen(z)
        xg = xg*.5+.5

        # Display generated images
        for j in range(num_pics_per_row):
            # Draw the reconstructed digits
            canvas_gen[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                xg[j].reshape([28, 28])
    

    createFigure(fs=(num_pics_per_row/2, num_pics_per_col/2))
    plt.imshow(canvas_gen, origin="upper", cmap="gray")
    plt.axis('off')
    plt.title("Fix codeword and vary only noise")
    
    for i in range(num_pics_per_col):
        z = sampleLatenVariable(num_pics_per_row, rand=np.random.randn(62))
        xg = DENGEN.getXGen(z)
        xg = xg*.5+.5

        # Display generated images
        for j in range(num_pics_per_row):
            # Draw the reconstructed digits
            canvas_gen[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                xg[j].reshape([28, 28])
    

    createFigure(fs=(num_pics_per_row/2, num_pics_per_col/2))
    plt.imshow(canvas_gen, origin="upper", cmap="gray")
    plt.axis('off')
    plt.title("Fix noise and vary only codeword")
    
      
    plt.show()