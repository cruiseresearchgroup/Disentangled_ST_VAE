# Disentangled_ST_VAE
This is the repo for paper "Measuring disentangled generative spatio-temporal representation" 

Sichen Zhao, Wei Shao, Jeffrey Chan, Flora D. Salim

Link to the paper : https://arxiv.org/abs/2202.04821

## Introduction
Disentangled representation learning offers useful properties such as dimension reduction and interpretability, which are essential to modern deep learning approaches. 
Although deep learning techniques have been widely applied to spatio-temporal data mining, there has been little attention to further disentangle the latent features and understanding their contribution to the model performance, particularly their mutual information and correlation across features.
In this study, we adopt two state-of-the-art disentangled representation learning methods and apply them to three large-scale public spatio-temporal datasets. To evaluate their performance, we propose an internal evaluation metric focusing on the degree of correlations among latent variables of the learned representations and the prediction performance of the downstream tasks. Empirical results show that our modified method can learn disentangled representations that achieve the same level of performance as existing state-of-the-art ST deep learning methods in a spatio-temporal sequence forecasting problem. Additionally, we find that our methods can be used to discover real-world spatial-temporal semantics to describe the variables in the learned representation.

## ARCHITECTURE

## Bibtex
If you find this code or the paper useful, please consider citing:
          
    @inproceedings{sichen2022measure,
      title={Measuring disentangled generative spatio-temporal representation},
      author={Zhao, Sichen and Shao, Wei and Chan, Jeffrey and Salim, Flora},
      booktitle={Proceedings of the 2022 SIAM International Conference on Data Mining (SDM)},
      pages={10--18},
      year={2022},
      organization={SIAM}
    }

## License

Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
