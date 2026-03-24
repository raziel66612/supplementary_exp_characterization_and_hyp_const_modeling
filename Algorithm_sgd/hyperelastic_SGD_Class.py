# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 13:12:50 2024

@author: avishwak
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class hyperelastic_sgd(nn.Module):
    def __init__(self,model,parameters,order, exp_strain, exp_stress):
        '''
        

        Parameters
        ----------
        model: (arg: string)
            DESCRIPTION: `Ogden` for now, but can be extended to other models in future.
        parameters: (arg: list)
            DESCRIPTION: Vector of initial values of the parameters of the model. The length of this vector should be consistent with the order of the model.
            e.g.(1): 2nd order Ogden = [μ1, μ2, α1, α2]
            e.g.(2): 8-chain = [μ, N]

        order: (arg: int)
            DESCRIPTION: Order of the model (1, 2, or 3 for Ogden)
        exp_strain: (arg: numpy.ndarray)
            DESCRIPTION: Experimental strain data
        exp_stress: (arg: numpy.ndarray)
            DESCRIPTION: Experimental stress data

        Returns
        -------
        None.

        '''

        self.model = model
        self.parameters = parameters
        self.order = order 
        self.exp_strain = exp_strain
        self.exp_stress = exp_stress
       
        if model == 'Ogden':
            if self.order == 3:
                μ1, μ2, μ3, α1, α2, α3 = self.parameters
                initial_guess_mu = μ1, μ2, μ3
                initial_guess_alpha  = α1, α2, α3 
                self.initial_guess_param=np.append(initial_guess_mu,initial_guess_alpha)
                self.nbparam = self.order*2

                print('Initial Guess : [μ1, μ2, μ3, α1, α2, α3] = ', self.initial_guess_param)
            elif self.order == 2:
                μ1, μ2, α1, α2 = self.parameters

                initial_guess_mu = μ1, μ2
                initial_guess_alpha = α1, α2
                self.initial_guess_param = np.append(initial_guess_mu, initial_guess_alpha)
                self.nbparam = self.order*2
                print('Initial Guess : [μ1, μ2, α1, α2] = ', self.initial_guess_param)
            
            elif self.order == 1:
                μ1, α1= self.parameters 
                
                initial_guess_mu = μ1
                initial_guess_alpha = α1
                self.initial_guess_param = np.append(initial_guess_mu, initial_guess_alpha)
                self.nbparam = self.order*2
                print('Initial Guess : [μ1,α1] = ', self.initial_guess_param)

            else:
                raise('incorrect vector size for arg "parameters" or "order" ')
        else:
            print('error')
            
            
        
    def OgdenModel(self, parameters, strain, loading_type = 'uniaxial' ):
        parameters = np.asarray(parameters)             # converting to numpy array, if it was a list
        self.mu_vec = parameters.reshape(2,self.order)[0]
        self.alpha_vec = parameters.reshape(2,self.order)[1]
        lammda=(strain)   

        #broadcast, using newaxis; to convert the given initial vector which is neither a row or a colm vector(as is in py), we convert it in colm vector. of shape, (order,1)
        lammda = lammda[np.newaxis,:]
        self.mu_vec = self.mu_vec[:self.order, np.newaxis]
        self.alpha_vec = self.alpha_vec[:self.order, np.newaxis]
        
        if loading_type == 'uniaxial':
            true_stress = np.sum(2*(self.mu_vec/self.alpha_vec)*(lammda**(self.alpha_vec) - lammda**(-(1/2)*self.alpha_vec)),axis = 0)  
            engg_stress = true_stress/lammda[:,1]
            tens = torch.tensor(engg_stress)
            return tens   #returning engineering strain for stretch, make sure abscissa are stretch and not strain
        
        elif loading_type == 'planar':
            true_stress = np.sum(2*(self.mu_vec/self.alpha_vec)*(lammda**(self.alpha_vec) - lammda**(-(1)*self.alpha_vec)),axis = 0)  
            engg_stress = true_stress/lammda[:,1]   #returning engineering strain for stretch, make sure abscissa are stretch and not strain
            tens = torch.tensor(engg_stress)
            return tens
        
        elif loading_type == 'biaxial':
            bi_stress = np.sum(2*(self.mu_vec/self.alpha_vec)*(lammda**(self.alpha_vec) - lammda**(-2*self.alpha_vec)),axis = 0)  
            true_stress = bi_stress/lammda[:,1]
            tens = torch.tensor(true_stress)
            return tens   #returning engineering strain for stretch, make sure abscissa are stretch and not strain
        
    
        
class Ogden_base_model(nn.Module):
    def __init__(self):
        super(Ogden_base_model, self).__init__()
        # self.initial_guess_param = torch.tensor(initial_guess_param)
        self.μ1 = nn.Parameter(0.1*torch.randn(1))
        self.μ2 = nn.Parameter(0.1*torch.randn(1))
        self.μ3 = nn.Parameter(0.1*torch.randn(1))
        self.α1 = nn.Parameter(0.1*torch.randn(1))
        self.α2 = nn.Parameter(0.1*torch.randn(1))
        self.α3 = nn.Parameter(0.1*torch.randn(1))
        
class Ogden_Order3(nn.Module):
    def __init__(self, base_model):
        super(Ogden_Order3, self).__init__()
        self.base_model = base_model
        
    def forward(self, stretch, order, loading_type ='uniaxial'):
        
        
        '''
        Define the forward model

        Parameters
        ----------
        stretch : Vector [Nx1]
            DESCRIPTION: This argument takes in a tensor of 1D for stretch
        order : Int (1,2,3)
            DESCRIPTION: set Ogden order 1,2,3
        loading_type : 'uniaxial', 'planar', 'biaxial'
            DESCRIPTION: The default is 'uniaxial'.
                        arg of 'planar', or 'biaxial' would invoke pure shear or biaxial equation.
        custom_param=None 
        custom_param: default is None; Inp arg is a vector.
             DESCRIPTION: If you wish to check the model (say Ogden 3), for a custom set of parameter
                          then provide a tensor vector of length 6 e.g. [0.1,0.1,0.1,0.01,0.02,0.03]
        Returns
        -------
        Theoretical stress vector
            returns engineering stress vector.

        '''
        
        
        self.loading_type = loading_type
        self.order = order
        # self.custom_param = custom_param
           
        
        #vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        if self.order == 3:
            self.mu_vec = torch.stack([self.base_model.μ1, self.base_model.μ2, self.base_model.μ3])
            self.alpha_vec = torch.stack([self.base_model.α1, self.base_model.α2, self.base_model.α3])
            
        elif self.order == 2:
            self.mu_vec = torch.stack([self.base_model.μ1, self.base_model.μ2])
            self.alpha_vec = torch.stack([self.base_model.α1, self.base_model.α2])
            self.nbparam = self.order*2
            
        elif self.order == 1:
            self.mu_vec = torch.stack([self.base_model.μ1])
            self.alpha_vec = torch.stack([self.base_model.α1])
            self.nbparam = self.order*2

        else:
            raise('incorrect vector size for arg "parameters" or "order" ')

        if self.loading_type == 'planar':
            stretch = torch.tensor(stretch, dtype=torch.float32)
            mu_vec_expanded = self.mu_vec.unsqueeze(0)
            # alpha_vec = torch.clamp(alpha_vec, min=1e-6)
            alpha_vec_expanded = self.alpha_vec.unsqueeze(0)
            true_stress = torch.sum(2 * (mu_vec_expanded / alpha_vec_expanded) * (stretch ** alpha_vec_expanded - stretch ** (-1 * alpha_vec_expanded)), dim=1)
            return true_stress/stretch[1]
        
        elif self.loading_type == 'biaxial':
            print('biaxial')
            return 
        
        stretch = torch.tensor(stretch, dtype=torch.float32)
        mu_vec_expanded = self.mu_vec.unsqueeze(0)
        # alpha_vec = torch.clamp(alpha_vec, min=1e-6)  # if we want to impose a constrain
        alpha_vec_expanded = self.alpha_vec.unsqueeze(0)
        true_stress = torch.sum(2 * (mu_vec_expanded / alpha_vec_expanded) * (stretch ** alpha_vec_expanded - stretch ** (-0.5 * alpha_vec_expanded)), dim=1)
        return true_stress/stretch[1]
    #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    
    def get_constrainted_param(self):
        mu1 = self.μ1
        mu2 = self.μ2
        mu3 = self.μ3
        alpha1 = (self.α1**2) * torch.sign(mu1 + 1e-6)
        alpha2 = (self.α2**2) * torch.sign(mu2 + 1e-6)
        alpha3 = (self.α3**2) * torch.sign(mu3 + 1e-6)
        return mu1, mu2, mu3, alpha1, alpha2, alpha3
    
    def constrained_param(self):
        return self.get_constrainted_param()

    print('Ogden3:tensor_inherited_class_in_work')
    
