'''
Created on Nov 10, 2018

@author: kyle
'''


import numpy as np
import tensorflow as tf


class Denoiser(object):
    def __init__(self, loc, input_tensor_name, output_tensor_name, 
                 input_tensor_shape, output_shape):
        '''
        loc: location of the trained model
        input_tensor_name: name of the neural network's input placeholder
        output_tensor_name: name of the neural network's output tensor
        input_tensor_shape: shape of the neural network's input placeholder
        output_shape: desired output shape
        '''
        self.graph = tf.Graph()
        self.sess = tf.Session(graph = self.graph)
        with self.graph.as_default():
            saver = tf.train.import_meta_graph(loc + '.meta', clear_devices=True)
            saver.restore(self.sess, loc)
            self.input = self.graph.get_tensor_by_name(input_tensor_name)
            self.input_shape = input_tensor_shape
            self.output_shape = output_shape
            self.output = self.graph.get_tensor_by_name(output_tensor_name)
            
#         print("Denoiser constructed")
    
    
    def denoise(self, x):
        '''
        Denoising by forward passing through the denoiser
        '''
        x = np.reshape(x, self.input_shape)
        x_star = self.sess.run(self.output, feed_dict={self.input: x})
        return np.reshape(x_star, self.output_shape)
    
    
class GenBasedDenoiser(Denoiser):
    '''
    Generative-model-based denoiser
    '''
    def __init__(self, loc, input_tensor_name, latent_tensor_name,
                 output_tensor_name, input_tensor_shape, output_shape,
                 isTrain_proj_name=None, isTrain_gen_name=None):
        super(GenBasedDenoiser, self).__init__(loc, input_tensor_name, output_tensor_name, 
                                               input_tensor_shape, output_shape)
        with self.graph.as_default():
            self.latent = self.graph.get_tensor_by_name(latent_tensor_name)
            if isTrain_proj_name is None:
                self.isTrain_proj_name = None
            else:
                self.isTrain_proj_name = self.graph.get_tensor_by_name(isTrain_proj_name)
            
            if isTrain_gen_name is None:
                self.isTrain_gen_name = None
            else:
                self.isTrain_gen_name = self.graph.get_tensor_by_name(isTrain_gen_name)
                
#         print("GenBasedDenoiser constructed")
        
        
    def denoise(self, x, return_latent_variable=False):
        '''
        Denoising by forward passing through the denoiser
        Also return the latent varialbe value if 
        return_latent_variable is set to True
        '''
        x = np.reshape(x, self.input_shape)
        if return_latent_variable:
            if self.isTrain_proj_name is None and self.isTrain_gen_name is None:
                x_star, z_star = self.sess.run([self.output, self.latent], feed_dict={self.input: x})
            elif self.isTrain_proj_name is None:
                x_star, z_star = self.sess.run([self.output, self.latent], feed_dict={self.input: x,
                                                                                      self.isTrain_gen_name: False})
            elif self.isTrain_gen_name is None:
                x_star, z_star = self.sess.run([self.output, self.latent], feed_dict={self.input: x,
                                                                                      self.isTrain_proj_name: False})
            else:
                x_star, z_star = self.sess.run([self.output, self.latent], feed_dict={self.input: x,
                                                                                      self.isTrain_gen_name: False,
                                                                                      self.isTrain_proj_name: False})
            
            return np.reshape(x_star, self.output_shape), z_star
        
        else:
            if self.isTrain_proj_name is None and self.isTrain_gen_name is None:
                x_star = self.sess.run(self.output, feed_dict={self.input: x})
            elif self.isTrain_proj_name is None:
                x_star = self.sess.run(self.output, feed_dict={self.input: x,
                                                               self.isTrain_gen_name: False})
            elif self.isTrain_gen_name is None:
                x_star = self.sess.run(self.output, feed_dict={self.input: x,
                                                               self.isTrain_proj_name: False})
            else:
                x_star = self.sess.run(self.output, feed_dict={self.input: x,
                                                               self.isTrain_gen_name: False,
                                                               self.isTrain_proj_name: False})
            
            return np.reshape(x_star, self.output_shape)
        
