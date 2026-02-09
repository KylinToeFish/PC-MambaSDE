import numpy as np
import torch

class IndustrialIrregularGenerator:
    def __init__(self, 
                 prob_sensor_drop=0.5,   
                 burst_prob=0.05,       
                 burst_length_mean=5,    
                 jitter_scale=0.1,     
                 noise_rul_factor=0.5,   
                 max_rul=125.0):        
        
        self.p_drop = prob_sensor_drop
        self.burst_p = burst_prob
        self.burst_mu = burst_length_mean
        self.jitter = jitter_scale
        self.alpha = noise_rul_factor
        self.max_rul = max_rul

    def transform(self, X, time_steps, RUL):

        if torch.is_tensor(X): X = X.numpy()
        if torch.is_tensor(time_steps): time_steps = time_steps.numpy()
        
        seq_len, feat_dim = X.shape
        Mask = np.ones_like(X)

        async_mask = np.random.choice([0, 1], size=(seq_len, feat_dim), p=[self.p_drop, 1 - self.p_drop])
        Mask *= async_mask

        t = 0
        while t < seq_len:
            if np.random.rand() < self.burst_p:
                length = int(np.random.poisson(self.burst_mu))
                length = max(1, length) 
                end = min(t + length, seq_len)
                
                if np.random.rand() < 0.5:
                    Mask[t:end, :] = 0  
                else:
                    feats_to_drop = np.random.choice(feat_dim, size=feat_dim//2, replace=False)
                    Mask[t:end, feats_to_drop] = 0
                
                t = end
            else:
                t += 1

        curr_rul = max(0, min(RUL, self.max_rul))
        degradation_factor = 1.0 - (curr_rul / self.max_rul)
        
        base_sigma = 0.01
        current_sigma = base_sigma * (1 + self.alpha * degradation_factor)
        
        noise = np.random.normal(0, current_sigma, size=X.shape)
        X_noisy = X + noise

        noise_t = np.random.normal(0, self.jitter, size=time_steps.shape)
        T_jittered = time_steps + noise_t
        
        T_jittered = np.sort(T_jittered)
        T_jittered = np.maximum(T_jittered, 0.0) 
        T_final = T_jittered / float(seq_len) 

        X_final = X_noisy * Mask
        
        return (torch.FloatTensor(X_final), 
                torch.FloatTensor(T_final), 
                torch.FloatTensor(Mask))
