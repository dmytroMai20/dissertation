import torch

class LinearNoise:
    def __init__(self, beta_start, beta_end, num_timesteps):
        self.beta_start= beta_start
        self.beta_end=beta_end
        self.num_timesteps = num_timesteps

        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1. - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1-self.alpha_bar)

    def forward_process(self, x_0, epsilon, t): # forward process return x_t from x_0 given time step and noise epsilon
        x_0_shape = x_0.shape
        batch_size = x_0_shape[0]

        sqrt_alpha_bar = self.sqrt_alpha_bar[t].reshape(batch_size)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar[t].reshape(batch_size)

        for _ in range(len(x_0_shape)-1):
            sqrt_alpha_bar = sqrt_alpha_bar.unsqueeze(-1)
        for _ in range(len(x_0_shape)-1):
            sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar.unsqueeze(-1)
        return (sqrt_alpha_bar.to(x_0.device)* x_0 + sqrt_one_minus_alpha_bar.to(x_0.device) * epsilon)
   
    def reverse_process(self, x_t, epsilon_pheta, t):
        x_0 = ((x_t - (self.sqrt_one_minus_alpha_bar.to(x_t.device)[t] * epsilon_pheta))/ torch.sqrt(self.alpha_bar.to(x_t.device)[t]))
        x_0 = torch.clamp(x_0, -1., 1.)

        mu = (x_t - ((self.betas.to(x_t.device)[t]) * epsilon_pheta) / (self.sqrt_one_minus_alpha_bar.to(x_t.device)[t])) / torch.sqrt(self.alphas.to(x_t.device)[t])

        if t == 0:
            return mu, x_0
        else:
            var = (1 - self.alpha_bar.to(x_t.device)[t-1]) / (1 - self.alpha_bar.to(x_t.device)[t])
            var = var * self.betas.to(x_t.device)[t]

            sigma = var ** 0.5
            epsilon = torch.randn(x_t.shape).to(x_t.device)
            return mu + sigma * epsilon, x_0