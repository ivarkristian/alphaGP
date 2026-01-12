import gpytorch.constraints
import torch
import gpytorch

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, type, lengthscale_constraint=gpytorch.constraints.Positive()):
        super().__init__(train_x, train_y, likelihood)
        
        self.max_losses = 10000
        self.losses = [0]*self.max_losses
        self.lengthscales = [0]*self.max_losses
        self.outputscales = [0]*self.max_losses
        self.iter = [0]*self.max_losses
        self.curr_trained = 0
        self.val_losses = [0]*self.max_losses
        self.val_iter = [0]*self.max_losses
        self.saved_val_losses = 0
        self.mix_losses = [0]*self.max_losses
        self.mix_iter = [0]*self.max_losses
        self.saved_mix_losses = 0
        self.add_noises = torch.tensor([0.0]*train_x.size(0))
        
        if type == 'scale_rbf':
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(lengthscale_constraint=lengthscale_constraint))
        elif type == 'rbf':
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.RBFKernel(lengthscale_constraint=lengthscale_constraint)
        elif type == 'scale_rq':
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel(lengthscale_constraint=lengthscale_constraint))
        elif type == 'scale_rbf_ard':
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=2, lengthscale_constraint=lengthscale_constraint))
        elif type == 'scale_matern':
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=0.5, lengthscale_constraint=lengthscale_constraint))
        elif type == 'matern':
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.MaternKernel(nu=0.5, lengthscale_constraint=lengthscale_constraint)
        elif type == 'matern_ard':
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=2, lengthscale_constraint=lengthscale_constraint)
        elif type == 'SMK':
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=8, ard_num_dims=2)
            self.covar_module.initialize_from_data(train_x, train_y)
        else:
            print(f"Error: Kernel type {type} not recognized.")

    def forward(self, x, add_training_noises=None, add_gating_noises=None):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        if add_training_noises is not None and add_gating_noises is not None:
            if covar_x.shape[0] == len(add_training_noises) + len(add_gating_noises):
                self.add_noises = torch.cat((add_training_noises, add_gating_noises))
                #print(f'adding resps and gating')
            elif covar_x.shape[0] == len(add_training_noises):
                #print(f'adding resps')
                self.add_noises = add_training_noises
            elif covar_x.shape[0] == len(add_gating_noises):
                self.add_noises = add_gating_noises
                #print(f'adding gating')    
            noise_matrix = torch.diag(self.add_noises)
            covar_x = covar_x + noise_matrix
        elif add_training_noises is not None:
            self.add_noises = add_training_noises
            noise_matrix = torch.diag(self.add_noises)
            covar_x = covar_x + noise_matrix
        
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    def save_loss(self, loss):
        i, ls = loss
        self.losses[self.curr_trained] = ls
        self.lengthscales[self.curr_trained] = self.get_lengthscale()
        self.outputscales[self.curr_trained] = self.get_outputscale()
        self.iter[self.curr_trained] = i
        self.curr_trained += 1
        return

    def save_val_loss(self, loss):
        i, ls = loss
        self.val_losses[self.saved_val_losses] = ls
        self.val_iter[self.saved_val_losses] = i
        self.saved_val_losses += 1
        return
    
    def save_mix_loss(self, loss):
        i, ls = loss
        self.mix_losses[self.saved_mix_losses] = ls
        self.mix_iter[self.saved_mix_losses] = i
        self.saved_mix_losses += 1
        return
    
    def print_named_parameters(self):
        for name, value in self.named_parameters():
            name_no_raw = name.replace('raw_', '')
            param = 'self.' + name_no_raw
            num_params = len(value.size())
            with torch.no_grad():
                res = eval(param)
                print(f'{name_no_raw}:', end='')
                if num_params == 0:
                    print(f' {res.item()}')
                else:
                    print(f' {res}')
    
    def get_lengthscale(self):
        for name, value in self.named_parameters():
            if 'lengthscale' in name:
                name_no_raw = name.replace('raw_', '')
                param = 'self.' + name_no_raw
                with torch.no_grad():
                    res = eval(param)
                    num_params = len(value.size())
                    if num_params == 0:
                        return res.item()
                    else:
                        return res[0]
        return False
    
    def get_outputscale(self):
        for name, value in self.named_parameters():
            if 'outputscale' in name:
                name_no_raw = name.replace('raw_', '')
                param = 'self.' + name_no_raw
                with torch.no_grad():
                    res = eval(param)
                    return res.item()

        return False