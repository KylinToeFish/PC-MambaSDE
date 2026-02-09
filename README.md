## Physically-Constrained Mamba-SDE for Remaining Useful Life Prediction under Irregular Observations
Accurate Remaining Useful Life prediction is critical for industrial predictive maintenance. However, real-world deployment is challenging due to the irregular nature of sensor observations, characterized by asynchronous sampling, burst missingness, and temporal jitter. Compounding this issue, purely data-driven models often generate physically implausible degradation trajectories that violate the irreversible nature of damage accumulation.
To address this, we propose PC-MambaSDE, a unified continuous-time framework for robust RUL prediction under irregular observations. Specifically, we design a Mask-Aware Continuous Mamba Encoder that explicitly leverages observation masks to extract context-rich control signals. Furthermore, we introduce a Physics-Guided Latent SDE with parametrically rectified hybrid drift, superimposing a global physical bias to enforce monotonic degradation even amid severe observation gaps. Additionally, we formulate RUL prediction as a boundary value problem via a Latent Degradation Bridge, which decouples a Health Index dimension and applies a Bridge Matching Loss to strictly constrain trajectories toward the failure state. Theoretically, we prove that our variational objective is mathematically equivalent to minimizing the KL divergence via Girsanov’s theorem, and we guarantee the global asymptotic stability of the learned dynamics through Lyapunov analysis.
To enable rigorous evaluation, we develop a Hybrid Irregularity Generation Scheme that simulates realistic industrial imperfections. Extensive experiments on public benchmarks demonstrate that PC-MambaSDE significantly outperforms state-of-the-art methods, particularly under extreme observation scarcity, validating the efficacy of embedding physical priors into continuous-time latent dynamics.


## Environment Details
```
python==3.8.8
numpy==1.20.1
pandas==1.2.4
matplotlib==3.3.4
pytorch==1.8.1
```
