# **Spectral bias in physics-informed and operator learning: Analysis and mitigation guidelines**

This repository contains the official implementations and datasets for the paper  
**"Spectral bias in physics-informed and operator learning: Analysis and mitigation guidelines"**  
by Siavash Khodakarami, Vivek Oommen, Nazanin Ahmadi Daryakenari, Maxim Beekenkamp, and George Em Karniadakis.

---

## Read the paper here:
[Spectral bias in physics-informed and operator learning: Analysis and mitigation guidelines](https://arxiv.org/abs/2602.19265)

---

## PyTorch Implementations

- PINN / PIKAN trained with Adam and SOAP for:
  - KdV equation
  - All four cases of the Wave equation
- All neural operators for:
  - Impinging jet problem
  - Earthquake problem

---

## JAX Implementations

- PINN / PIKAN trained with SS-Broyden for:
  - KdV equation
  - All four cases of the Wave equation
- PINN / PIKAN for steady-state diffusion-reaction equation:
  - All optimizers

---

## Citing This Work
If you use this work, please cite:

```bibtex
@article{khodakarami2026spectral,
  title={Spectral bias in physics-informed and operator learning: Analysis and mitigation guidelines},
  author={Khodakarami, Siavash and Oommen, Vivek and Daryakenari, Nazanin Ahmadi and Beekenkamp, Maxim and Karniadakis, George Em},
  journal={arXiv preprint arXiv:2602.19265},
  year={2026}
}
```

---

## Acknowledgements

We acknowledge the following repositories:

1. KdV ground truth data (numerical simulation): https://github.com/PredictiveIntelligenceLab/jaxpi/tree/pirate
2. SS-Broyden optimizer JAX implementation: https://github.com/jdtoscano94/NABLA-SciML/tree/main/vRBA_variational_residual_based_attention_PINNs_Operator_learning 
