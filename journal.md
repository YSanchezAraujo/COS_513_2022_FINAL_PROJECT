# Jan 30, 2022 
I created this repo for the final project. Much to do

# April 4th, 20202
At this point I have an HMM model implemented, now I'm figuring how the additional steps required for the HMM-GLM model. For the basic HMM model with Gaussian emissions we have that the means $\mu$ are given by: 

$$\mu_k = p(z_k)^{\top} y$$

I'm wondering if, given thhe above the following holds: 

$$\mu_k = X w_k$$

$$X w_k = p(z_k)^{\top} y$$

$$w_k = (X^{\top}X)^{-1}X^{\top} p(z_k)^{\top} y$$
