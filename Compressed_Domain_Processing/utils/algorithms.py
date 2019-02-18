'''
Created on Nov 10, 2018

@author: kyle
'''


import numpy as np
import scipy.sparse.linalg as linalg
from copy import deepcopy as deepcopy


class PnPADMM(object):
    def __init__(self, (M, N), algo_param=None):
        '''
        (M, N): M - number of measurements, N - original dimension
        algo_param: parameter dictionary containing:
            "rho": parameter to control the constraint x = z
            "x0": starting point all the algorithm
            "tol": allowance for number of non-loss-decreasing iterations
            "maxiter": maximum number of iterations
            "callback": call back function, called after each solving iteration
                        must take input in the order of: x, z, u, loss
        '''
        self.shape = (M, N)
        if algo_param is None:
            self.algo_param = {}
        else:
            self.algo_param = deepcopy(algo_param)
            
        if "rho" not in self.algo_param:
            self.algo_param["rho"] = 1e2
            
        if "x0" not in self.algo_param:
            self.algo_param["x0"] = np.random.randn(N)
            
        if "tol" not in self.algo_param:
            self.algo_param["tol"] = 50
        
        if "maxiter" not in self.algo_param:
            self.algo_param["maxiter"] = 100
            
        if "callback" not in self.algo_param:
            self.algo_param["callback"] = None
                 
        
    def solve(self, y, A, Denoiser, cg_param=None):
        '''
        Use the plug-and-play ADMM algorithm for compressive sensing recovery
        For the sub-least-square step:
            conjugate gradient method is used when A is given as a linear operator
            direct inverse is calcuated when A is given as a matrix 
        Input:
            y: numpy array of shape (M,)
            A: numpy matrix of shape (M, N) or scipy.sparse.linalg.LinearOperator
            Denoiser: Denoiser class (see module import_neural_networks), must have vector output
            cg_param: parameter for scipy.sparse.linalg.cg
        Reture:
            x_star: recovered signal of shape (N,)
            info: information stored from callback functions
        '''
        if isinstance(A,linalg.LinearOperator):
            # copy cg_param
            if cg_param is None:
                self.cg_param = {}
            else:
                self.cg_param = deepcopy(cg_param)
                
            if "tol" not in self.cg_param:
                self.cg_param["tol"] = 1e-5
            
            if "maxiter" not in self.cg_param:
                self.cg_param["maxiter"] = None
                
            # Build new linear operators for cg
            P_mv = lambda x: A.rmatvec(A.matvec(x)) + self.algo_param["rho"]*x
            P = linalg.LinearOperator((self.shape[1], self.shape[1]), matvec=P_mv, rmatvec=P_mv)
            q = A.rmatvec(y)
            # Initial iterations
            loss_func = lambda x: np.sum(np.square(y - A.matvec(x)))  # define objective function
            loss = loss_func(self.algo_param["x0"])
            loss_star = loss
            x = self.algo_param["x0"]
            z = x
            u = np.zeros_like(x)
            k = 0
            tol = self.algo_param["tol"]
            loss_record = np.array([])
            z_record = []
            callback_res = []
            # Start iterations
            while k < self.algo_param["maxiter"] and tol > 0:
                # least square step
                x, _ = linalg.cg(P, q + self.algo_param["rho"]*(z - u), x0=z, 
                                 tol=self.cg_param["tol"], maxiter=self.cg_param["maxiter"])
                # denoising step
                z = Denoiser.denoise(x + u)
                # dual variable update
                u += x - z
                # monitor the loss
                loss =  loss_func(z)
                if loss < loss_star:
                    loss_star = loss
                else:
                    tol -= 1
                    
                loss_record = np.append(loss_record, loss)
                # record all the denoised signals
                z_record.append(z)
                # callback functions
                if self.algo_param["callback"] is not None:
                    callback_res.append(self.algo_param["callback"](x, z, u, loss))
                    
                k += 1
         
            x_star = z_record[np.argmin(loss_record)]
            
        else:      
            # One time calculation
            P = np.linalg.inv(A.T.dot(A) + self.algo_param["rho"]*np.eye(self.shape[1]))
            q = P.dot(A.T.dot(y))
            # Initial iterations
            loss_func = lambda x: np.sum(np.square(y - A.dot(x)))  # define objective function
            loss = loss_func(self.algo_param["x0"])
            loss_star = loss
            x = self.algo_param["x0"]
            z = x
            u = np.zeros_like(x)
            k = 0
            tol = self.algo_param["tol"]
            loss_record = np.array([])
            z_record = []
            callback_res = []
            # Start iterations
            while k < self.algo_param["maxiter"] and tol > 0:
                # least square step
                x = q +  self.algo_param["rho"]*P.dot(z-u)
                # denoising step
                z = Denoiser.denoise(x + u)
                # dual variable update
                u += x - z
                # monitor the loss
                loss =  loss_func(z)
                if loss < loss_star:
                    loss_star = loss
                else:
                    tol -= 1
                    
                loss_record = np.append(loss_record, loss)
                # record all the denoised signals
                z_record.append(z)
                # callback functions
                if self.algo_param["callback"] is not None:
                    callback_res.append(self.algo_param["callback"](x, z, u, loss))
                    
                k += 1
         
            x_star = z_record[np.argmin(loss_record)]
        
        return x_star, callback_res
    
