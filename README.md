# GID

- This is the source code of "Y. Zhang, C. Yang, K. Huang and Y. Li, "Intrusion Detection of Industrial Internet-of-Things Based on Reconstructed Graph Neural Networks," in IEEE Transactions on Network Science and Engineering, 2022, doi: 10.1109/TNSE.2022.3184975." 

- Mississippi data, mentioned in the paper, can be obtained from [Mississippi Industrial Control System (ICS) Cyber Attack Datasets](https://sites.google.com/a/uah.edu/tommy-morris-uah/ics-data-sets)

- The data in [/data/HardinLoopPlatform](https://github.com/MrZhangCSU/GID/tree/main/data/HardinLoopPlatform) is collected from the hard-in-the-loop platform of School of Automation of Central South University.
The photo of the self-built platform is shown below:
<div align=center><img src="https://raw.githubusercontent.com/MrZhangCSU/GID/main/HILP.png"></div>
<div align=center>This is the HILP platform we built </div>

- The DOI of this paper is: [10.1109/TNSE.2022.3184975](https://doi.org/10.1109/TNSE.2022.3184975)

- If you find the source code useful, please cite:
```latex
@ARTICLE{9802721,
  author={Zhang, Yichi and Yang, Chunhua and Huang, Keke and Li, Yonggang},
  journal={IEEE Transactions on Network Science and Engineering}, 
  title={Intrusion Detection of Industrial Internet-of-Things Based on Reconstructed Graph Neural Networks}, 
  year={2023},
  volume={10},
  number={5},
  pages={2894-2905},
  abstract={Industrial Internet-of-Things (IIoT) are highly vulnerable to cyber-attacks due to their open deployment in unattended environments. Intrusion detection is an efficient solution to improve security. However, because the labeled samples are difficult to obtain, and the sample categories are imbalanced in real applications, it is difficult to obtain a reliable model. In this paper, a general framework for intrusion detection is proposed based on graph neural network technologies. In detail, a network embedding feature representation is proposed to deal with the high dimensional, redundant but categories imbalanced and rare labeled data in IIoT. To avoid the influence caused by the inaccurate network structure, a network constructor with refinement regularization is designed to amend it. At last, the network embedding representation weights and network constructor are trained together. The high accuracy and robust properties of the proposed method were verified by conducting intrusion detection tasks based on public datasets. Compared with several state-of-art algorithms, the proposed framework outperforms these methods in many evaluation metrics. In addition, a hard-in-the-loop platform is designed to test the performance in real environments. The results show that the method can not only identify different attacks but also distinguish between cyber-attacks and physical failures.},
  keywords={},
  doi={10.1109/TNSE.2022.3184975},
  ISSN={2327-4697},
  month={Sep.},}

```
