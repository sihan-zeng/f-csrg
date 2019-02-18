'''
Created on Nov 11, 2018

@author: kyle
'''
import sys
sys.path.insert(0, '../../utils')
import numpy as np
import tensorflow as tf
import scipy.sparse.linalg as linalg
import matplotlib.pyplot as plt
from algorithms import PnPADMM
from import_neural_networks import Denoiser, GenBasedDenoiser


def constructDenoisers():
    # DAE-based denoiser
    model_loc = "../../../Neural_Networks/MNIST/DAE_Vincent/trained_model/iter-90000"
    input_tensor_name = 'X:0'
    output_tensor_name = 'XRec:0'
    input_tensor_shape = (1, 784)
    output_shape = (784)
    denoiser_DAE = Denoiser(model_loc, input_tensor_name, output_tensor_name, 
                            input_tensor_shape, output_shape)
    print "DAE-based denoiser constructed"
    
    # DCGAN-based denoiser
    model_loc = "../../../Neural_Networks/MNIST/DCGAN/trained_model_dn/epoch-199"
    input_tensor_name = 'Xnoisy:0'
    output_tensor_name = 'Xdn:0'
    latent_tensor_name = 'Z:0'
    input_tensor_shape = (1, 28, 28, 1)
    output_shape = (784)
    isTrain_proj_name = 'isTrain_Proj:0'
    isTrain_gen_name = 'isTrain_Gen:0'
    denoiser_DCGAN = GenBasedDenoiser(model_loc, input_tensor_name, latent_tensor_name,
                                      output_tensor_name, input_tensor_shape, output_shape,
                                      isTrain_proj_name, isTrain_gen_name)
    print "DCGAN-based denoiser constructed"
    
    # InfoGAN-based denoiser
    model_loc = "../../../Neural_Networks/MNIST/InfoGAN/trained_model_dn/epoch-499"
    input_tensor_name = 'Xnoisy:0'
    output_tensor_name = 'Xdn:0'
    latent_tensor_name = 'Proj:0'
    input_tensor_shape = (1, 28, 28, 1)
    output_shape = (784)
    isTrain_proj_name = 'isTrain_Proj:0'
    isTrain_gen_name = 'isTrain_Gen:0'
    denoiser_InfoGAN = GenBasedDenoiser(model_loc, input_tensor_name, latent_tensor_name,
                                        output_tensor_name, input_tensor_shape, output_shape,
                                        isTrain_proj_name, isTrain_gen_name)
    print "InfoGAN-based denoiser constructed"
    
    return {"DAE": denoiser_DAE, "DCGAN": denoiser_DCGAN, "InfoGAN": denoiser_InfoGAN}


def setAlgoParams():
    algo_param_DAE = {"rho": 10 / np.sqrt(M),
                      "x0": np.zeros(784),
                      "tol": 150,
                      "maxiter": 200,
                      "callback": None
    }
    algo_param_DCGAN = {"rho": 10 / np.sqrt(M),
                        "x0": np.zeros(784),
                        "tol": 150,
                        "maxiter": 200,
                        "callback": None
    }
    algo_param_InfoGAN = {"rho": 10 / np.sqrt(M),
                          "x0": np.zeros(784),
                          "tol": 150,
                          "maxiter": 200,
                          "callback": None
    }
    
    return {"DAE": algo_param_DAE, "DCGAN": algo_param_DCGAN, "InfoGAN": algo_param_InfoGAN}


def getTestImages(n=1, lb_target=None, rand_shuffle=True):
    mnist = tf.keras.datasets.mnist
    (_, _), (test_set, lb) = mnist.load_data()
    test_set = np.float32(test_set) / 255.0
    if rand_shuffle:
        idx_all = np.arange(len(test_set))
        np.random.shuffle(idx_all)
        test_set = test_set[idx_all]
        lb = lb[idx_all]
    
    if lb_target is None:
        x_true = test_set[:n].reshape([n, 784])
        lb_true = lb[:n].reshape(n)
        return x_true, lb_true
    else:
        x_true = np.zeros([n, 784])
        lb_true = np.zeros(n)
        k = 0
        for idx, l in enumerate(lb):
            if l == lb_target:
                x_true[k] = test_set[idx].reshape(784)
                lb_true[k] = lb[idx]
                k += 1
                
            if k == n:
                break
            
        return x_true[:k], lb_true[:k]


if __name__ == '__main__':
    # Program parameters
    show_plot = True  #False
    save_results = False  #True
    sparse_A = False
    n_test = 5  #10000
    comp_ratios = [32]  #[4, 8, 16, 32, 64] 
    
    # Get test set
    test_imgs, test_lbs = getTestImages(n_test)
    
    # Construct denoisers
    denoisers_dict = constructDenoisers()
    
    # dictionary storing all the recovered results
    results_x = []
    results_lb = []
    results_xr = {}
    
    for comp_ratio in comp_ratios:
        for i in xrange(n_test):
            print "----------- CS Ratio", str(comp_ratio)+"x,",  " Iteration", i+1, "-----------"
            
            # Access to MNIST dataset and choose one test image
            lb_target = None
            x_true = test_imgs[i].reshape(784)
            x_true_rescaled = x_true*2 - 1
            results_x.append(x_true.reshape([28, 28]))
            results_lb.append(test_lbs[i])
                
            # CS setups
            N = 784  # full signal dimension
            M = N/comp_ratio  # number of compressed measurements
            if sparse_A:
                A_prob = np.random.rand(M, N)
                p = .1
                A = np.zeros([M, N])
                A[A_prob < p/2.] = -1 / np.sqrt(M) / p
                A[A_prob > 1-p/2.] = 1 / np.sqrt(M) / p
            else:
                A = np.random.randn(M, N) / np.sqrt(M)
            
            Aopt = linalg.LinearOperator((M, N), matvec=lambda x: A.dot(x),
                                         rmatvec=lambda x: A.T.dot(x))
            y = Aopt.matvec(x_true)
            y_rescaled = Aopt.matvec(x_true_rescaled)
            
            # Construct a PnPADMM solver
            algo_params_dict = setAlgoParams()
            
            # Solve
            x_res = {}
            for denoiser_name in denoisers_dict:
                denoiser = denoisers_dict[denoiser_name]
                solver = PnPADMM((M, N), algo_params_dict[denoiser_name])
                if isinstance(denoiser, GenBasedDenoiser):
                    x_star, _ = solver.solve(y_rescaled, A, denoiser)
                    x_star = x_star*.5 + .5
                else:
                    x_star, _ = solver.solve(y, A, denoiser)
                    
                x_res[denoiser_name] = x_star
                if denoiser_name not in results_xr:
                    results_xr[denoiser_name] = [x_star.reshape([28, 28])]
                else:
                    results_xr[denoiser_name].append(x_star.reshape([28, 28]))
                    
                print "L2 error"+denoiser_name+": ", np.linalg.norm(x_star-x_true)
            
            # Show results
            if show_plot and n_test <= 10:
                plt.figure()
                plt.subplot(1,len(denoisers_dict)+1,1)
                plt.imshow(x_true.reshape([28, 28]), origin="upper", cmap="gray")
                plt.axis('off')
                plt.title(str(comp_ratio)+"x Compression\n Original")
                i = 2
                for denoiser_name in denoisers_dict:
                    plt.subplot(1,len(denoisers_dict)+1,i)
                    plt.imshow(x_res[denoiser_name].reshape([28, 28]), origin="upper", cmap="gray")
                    plt.title(denoiser_name)
                    plt.axis('off')
                    i += 1
        
        # Save results
        if save_results:        
            np.save("method_comparison_results/results_x_"+str(comp_ratio)+"x", results_x)
            np.save("method_comparison_results/results_lb_"+str(comp_ratio)+"x", results_lb)
            np.save("method_comparison_results/results_xr_"+str(comp_ratio)+"x", results_xr)
            results_x = []
            results_lb = []
            results_xr = {}
            
    plt.show()
        
        