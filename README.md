## [Explicit physics-informed neural networks for nonlinear closure: The case of transport in tissues](https://github.com/TaghizadehE/explicit-PINN)

data-driven machine learning for explicitly closing nonlinear problems during upscaling.

We use data-driven machine learning for explicitly closing nonlinear problems during upscaling.  We focus on the problem of upscaling the differential balance for convection, diffusion, and nonlinear reaction in biological tissues.  The classical effectiveness factor model is used to formulate the macroscale reaction kinetics. The network is described as being _explicit_ in the sense that the network is trained using macroscale concentrations and gradients of concentration as components of the feature space. The upscaled solutions for the average concentration are compared with numerical solutions derived from the microscale concentration fields by a posteriori averaging.
There are three outcomes of this work of particular note: 1) we find that that the trained network exhibits good generalizability, and it is able to predict the effectiveness factor with high fidelity for realistically-structured tissues despite the significantly different scale and geometry of the two example tissue types; 2) the approach results in an upscaled PDE with an effectiveness factor that is predicted (implicitly) via the trained neural network; and 3) more importantly, the implicit features derived from the source term appearing in the microscale closure problem are essential for the network to predict the correction factor with high fidelity.

For more information, please refer to the following.

- Ehsan Taghizadeh, Helen M. Byrne, Brian D. Wood. ["Explicit physics-informed neural networks for non-linear upscaling closure: the case of transport in tissues."](https://www.sciencedirect.com/science/article/pii/S0021999121006768) Journal of Computational Physics, 110781 (2021).

## **Citation**
```
@article{taghizadeh2021explicit,
  title={Explicit physics-informed neural networks for nonlinear closure: The case of transport in tissues},
  author={Taghizadeh, Ehsan and Byrne, Helen and Wood, Brian D},
  journal={Journal of Computational Physics},
  pages={110781},
  year={2021},
  publisher={Elsevier}
}

@article{taghizadeh2021explicit,
         title={Explicit physics-informed neural networks for non-linear upscaling closure: the case of transport in tissues}, 
         author={Ehsan Taghizadeh and Helen M. Byrne and Brian D. Wood},
         journal={arXiv preprint arXiv:2104.01476},
         year={2021},
}
```

## **Instruction**
1. Run _run.py_ for training.
2. Run _CS_SS_CDR_DN_ML.py_ to merge learned correction factor with central space steady state convection diffusion Michaelis–Menten–Monod kinetics reaction left dirichlet right neumann finite difference.
