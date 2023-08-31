# MLIC Series [ACMMM 2023 / NCW ICML 2023]

Release the code of MLIC <sup> ++ </sup> ! 

We highlight **MLIC <sup> ++ </sup>**, which **sloves the quadratic complexity of global context capturing**!

*MLIC: Multi-Reference Entropy Model for Learned Image Compression* [[Arxiv](https://arxiv.org/abs/2211.07273)] is accepted at ACMMM 2023 !

*MLIC <sup> ++ </sup>: Linear Complexity Multi-Reference Entropy Modeling for Learned Image Compression*  [[Arxiv](https://arxiv.org/abs/2307.15421)] [[OpenReview](https://openreview.net/forum?id=hxIpcSoz2t)] is accepted at ICML 2023 Neural Compression Workshop !

## Architectures

![image](assets/arch.png)


## Performance
![image](assets/kodak.png)
![image](assets/tecnick.png)
![image](assets/clicp.png)

## Pretrained Models

#### Bug Fixes

I fix the implementation of *LinearGlobalIntraContext*.

New pre-trained models will be released soon. The performance is slightly better than before on Kodak and Tecnick.

#### New Weights

| Lambda | Metric | Link |
|--------|--------|------|
| 0.0018   | MSE    |   [PKUDisk](https://disk.pku.edu.cn:443/link/56ABCF09A715A197523E5B8929DBA2BB)   |
| 0.0035   | MSE    |   [PKUDisk](https://disk.pku.edu.cn:443/link/22775C0DBB903AC3A43342C8AFDBFD05)   |
| 0.0067   | MSE    |   [PKUDisk](https://disk.pku.edu.cn:443/link/9474C67EE30DCCB3C77CDCC459425B38)   |
| 0.0130   | MSE    |   [PKUDisk](https://disk.pku.edu.cn:443/link/59F4117444A787B253DE04D72C4AE2AB)   |
| 0.0250   | MSE    |   [PKUDisk](https://disk.pku.edu.cn:443/link/00200D4B21E7428471DFF69C5B9878E5)   |

Training details: We train each model on a single Tesla A100 GPU. The batch size is set to $32$. The initial 
patch size is set to $256\times 256$. We set the patch size to $512\times 512$ after $1.2$ M steps. 

## Environment

CompressAI 1.2.0b3

## Contact

If you have any questions about MLIC, please contact Wei Jiang ( wei.jiang1999@outlook.com or jiangwei@stu.pku.edu.cn )

## Citation
```
@article{jiang2022mlic,
  title={Multi-reference entropy model for learned image compression},
  author={Jiang, Wei and Yang, Jiayu and Zhai, Yongqi and Wang, Ronggang},
  journal={arXiv preprint arXiv:2211.07273},
  year={2022}
}
```

```
@article{jiang2023mlic,
  title={MLIC++: Linear Complexity Multi-Reference Entropy Modeling for Learned Image Compression}, 
  author={Jiang, Wei and Wang, Ronggang},
  journal={arXiv preprint arXiv:2307.15421},
  year={2023},
}
```
