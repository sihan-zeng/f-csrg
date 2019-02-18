'''
Created on Feb 15, 2019

@author: kyle
'''

import sys
sys.path.insert(0, '../../utils')
import numpy as np
import scipy.sparse.linalg as linalg
import matplotlib.pyplot as plt
from algorithms import PnPADMM
from import_neural_networks import Denoiser, GenBasedDenoiser


def constructDenoisers():
    # DAE-based denoiser
    model_loc = "../../../Neural_Networks/CelebA/DAE/trained_AE_dn/epoch-499"
    input_tensor_name = 'Xnoisy:0'
    output_tensor_name = 'Xdn:0'
    input_tensor_shape = (1, 32, 32, 1)
    output_shape = (1024)
    denoiser_DAE = Denoiser(model_loc, input_tensor_name, output_tensor_name, 
                            input_tensor_shape, output_shape)
    print "DAE-based denoiser constructed"
    
    # DCGAN-based denoiser
    model_loc = "../../../Neural_Networks/CelebA/DCGAN/trained_model_dn_dcgan/epoch-109"
    input_tensor_name = 'Xnoisy:0'
    output_tensor_name = 'Xdn:0'
    latent_tensor_name = 'Proj:0'
    input_tensor_shape = (1, 32, 32, 1)
    output_shape = (1024)
    isTrain_proj_name = 'isTrain_Proj:0'
    isTrain_gen_name = 'isTrain_Gen:0'
    denoiser_DCGAN = GenBasedDenoiser(model_loc, input_tensor_name, latent_tensor_name,
                                      output_tensor_name, input_tensor_shape, output_shape,
                                      isTrain_proj_name, isTrain_gen_name)
    print "DCGAN-based denoiser constructed"
     
    # InfoGAN-based denoiser
    model_loc = "../../../Neural_Networks/CelebA/InfoGAN/trained_model_dn_infogan/epoch-499"
    input_tensor_name = 'Xnoisy:0'
    output_tensor_name = 'Xdn:0'
    latent_tensor_name = 'Proj:0'
    input_tensor_shape = (1, 32, 32, 1)
    output_shape = (1024)
    isTrain_proj_name = 'isTrain_Proj:0'
    isTrain_gen_name = 'isTrain_Gen:0'
    denoiser_InfoGAN = GenBasedDenoiser(model_loc, input_tensor_name, latent_tensor_name,
                                        output_tensor_name, input_tensor_shape, output_shape,
                                        isTrain_proj_name, isTrain_gen_name)
    print "InfoGAN-based denoiser constructed"
     
    return {"DAE": denoiser_DAE, "DCGAN": denoiser_DCGAN, "InfoGAN": denoiser_InfoGAN}


def setAlgoParams():
    algo_param_DAE = {"rho": 10 / np.sqrt(M),
                      "x0": np.zeros(1024),
                      "tol": 150,
                      "maxiter": 200,
                      "callback": None
    }
    algo_param_DCGAN = {"rho": 10 / np.sqrt(M),
                        "x0": np.zeros(1024),
                        "tol": 150,
                        "maxiter": 200,
                        "callback": None
    }
    algo_param_InfoGAN = {"rho": 10 / np.sqrt(M),
                          "x0": np.zeros(1024),
                          "tol": 150,
                          "maxiter": 200,
                          "callback": None
    }
     
    return {"DAE": algo_param_DAE, "DCGAN": algo_param_DCGAN, "InfoGAN": algo_param_InfoGAN}


def getTestImages(n=1, rand_shuffle=True):
    test_set = np.load('../../../Datasets/CelebA/Test.npy')
    if rand_shuffle:
        idx_all = np.arange(len(test_set))
        np.random.shuffle(idx_all)
        test_set = test_set[idx_all]
    
    if n > len(test_set):
        n = len(test_set)
        
    x_true = test_set[:n].reshape([n, 1024])     
    return x_true


if __name__ == '__main__':
    # Program parameters
    show_plot = True
    save_results = False
    sparse_A = False
    n_test = 2
    comp_ratios = [4, 8, 16, 32] 
    
    # Get test set
    test_imgs = getTestImages(n_test)
    
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
            x_true = test_imgs[i].reshape(1024)
            results_x.append(x_true.reshape([32, 32]))
                
            # CS setups
            N = 1024  # full signal dimension
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
            
            # Construct a PnPADMM solver
            algo_params_dict = setAlgoParams()
            
            # Solve
            x_res = {}
            for denoiser_name in denoisers_dict:
                denoiser = denoisers_dict[denoiser_name]
                solver = PnPADMM((M, N), algo_params_dict[denoiser_name])
                x_star, _ = solver.solve(y, A, denoiser)
                    
                x_res[denoiser_name] = x_star
                if denoiser_name not in results_xr:
                    results_xr[denoiser_name] = [x_star.reshape([32, 32])]
                else:
                    results_xr[denoiser_name].append(x_star.reshape([32, 32]))
                    
                print "L2 error"+denoiser_name+": ", np.linalg.norm(x_star-x_true)
            
            # Show results
            if show_plot and n_test <= 10:
                plt.figure()
                plt.subplot(1,len(denoisers_dict)+1,1)
                plt.imshow(x_true.reshape([32, 32]), origin="upper", cmap="gray")
                plt.axis('off')
                plt.title(str(comp_ratio)+"x Compression\n Original")
                i = 2
                for denoiser_name in denoisers_dict:
                    plt.subplot(1,len(denoisers_dict)+1,i)
                    plt.imshow(x_res[denoiser_name].reshape([32, 32]), origin="upper", cmap="gray")
                    plt.title(denoiser_name)
                    plt.axis('off')
                    i += 1
        
        # Save results
        if save_results:        
            np.save("method_comparison_results/results_x_"+str(comp_ratio)+"x", results_x)
            np.save("method_comparison_results/results_xr_"+str(comp_ratio)+"x", results_xr)
            results_x = []
            results_xr = {}
            
    plt.show()
        
        