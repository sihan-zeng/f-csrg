'''
Created on Feb 7, 2019

@author: kyle
'''


import numpy as np
import tensorflow as tf
import h5py  #@UnusedImport


if __name__ == '__main__':
    save_path = '../../../../Neural_Networks/MNIST/CNN_Classifier/trained_model/epoch-10.h5'
    model = tf.keras.models.load_model(save_path)
    
    for cr in [4, 8, 16, 32, 64]:
        print "-----------------", str(cr)+"x Compression", "-----------------"
        results_x = np.load("results_x_"+str(cr)+"x.npy")
        results_lb = np.load("results_lb_"+str(cr)+"x.npy")
        results_xr = np.load("results_xr_"+str(cr)+"x.npy").item()
        
        res = model.evaluate(results_x, results_lb, verbose=0)
        print "Classification accuracy (Original):", res[1]
        print "--------------------\n"
        
        for denoiser_name in results_xr:
            l2_err = np.zeros(len(results_x))
            for k in xrange(len(results_x)):
                l2_err[k] = np.linalg.norm(results_xr[denoiser_name][k] - results_x[k])
            
            print "L2 Error", denoiser_name+":", np.mean(l2_err)
    
            res = model.evaluate(np.asarray(results_xr[denoiser_name]), results_lb, verbose=0)
            print "Classification accuracy (Recovered)", denoiser_name+":", res[1]
                 
            print "--------------------\n"