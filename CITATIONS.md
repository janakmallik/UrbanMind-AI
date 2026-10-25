# Citations and Credits

All papers, datasets, models, and libraries used in UrbanMind-AI.

## Datasets

### NYC 311 Service Requests
NYC Open Data. *311 Service Requests from 2010 to Present*.
Dataset: https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2010-to-Present/erm2-nwe9
License: NYC Open Data Terms of Use (public).

### METR-LA
Li, Y., Yu, R., Shahabi, C., & Liu, Y. (2018).
*Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting*.
International Conference on Learning Representations (ICLR).
arXiv: https://arxiv.org/abs/1707.01926
Repository: https://github.com/liyaguang/DCRNN
License: Research use (see original paper).

### BDD100K
Yu, F., Chen, H., Wang, X., Xian, W., Chen, Y., Liu, F., Madhavan, V., & Darrell, T. (2020).
*BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning*.
The IEEE Conference on Computer Vision and Pattern Recognition (CVPR).
Paper: https://arxiv.org/abs/1805.04687
Official site: https://bdd-data.berkeley.edu/
License: UC Berkeley Regents License (research use).

### HMDB-51
Kuehne, H., Jhuang, H., Garrote, E., Poggio, T., & Serre, T. (2011).
*HMDB: A Large Video Database for Human Motion Recognition*.
IEEE International Conference on Computer Vision (ICCV), 2556-2563.
DOI: https://doi.org/10.1109/ICCV.2011.6126543
Repository: https://huggingface.co/datasets/Serrelab/hmdb51
License: CC BY 4.0.

### UCF-Crime
Sultani, W., Chen, C., & Shah, M. (2018).
*Real-world Anomaly Detection in Surveillance Videos*.
IEEE Conference on Computer Vision and Pattern Recognition (CVPR).
arXiv: https://arxiv.org/abs/1801.04264
Repository: https://www.crcv.ucf.edu/projects/real-world/
License: Research use (see original paper).

## UrbanSound8K

Salamon, J., Jacoby, C., & Bello, J. P. (2014). *A Dataset and Taxonomy
for Urban Sound Research*. Proceedings of the 22nd ACM International
Conference on Multimedia (MM '14), 1041–1044.
https://doi.org/10.1145/2647868.2655045

Accessed via the HuggingFace parquet mirror (`ml-data/urbansound8k`),
8,732 clips across 10 urban sound classes, 10 predefined CV folds.

## Models / Architectures

### ResNet
He, K., Zhang, X., Ren, S., & Sun, J. (2016).
*Deep Residual Learning for Image Recognition*.
IEEE Conference on Computer Vision and Pattern Recognition (CVPR).
arXiv: https://arxiv.org/abs/1512.03385

### DistilBERT
Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019).
*DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter*.
arXiv: https://arxiv.org/abs/1910.01108

### SegFormer
Xie, E., Wang, W., Yu, Z., Anandkumar, A., Alvarez, J. M., & Luo, P. (2021).
*SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers*.
Advances in Neural Information Processing Systems (NeurIPS).
arXiv: https://arxiv.org/abs/2105.15203

### YOLOv8
Jocher, G., Chaurasia, A., & Qiu, J. (2023).
*Ultralytics YOLOv8* (Version 8.0.0) [Computer software].
https://github.com/ultralytics/ultralytics
License: AGPL-3.0.

### Graph WaveNet
Wu, Z., Pan, S., Long, G., Jiang, J., & Zhang, C. (2019).
*Graph WaveNet for Deep Spatial-Temporal Graph Modeling*.
International Joint Conference on Artificial Intelligence (IJCAI).
arXiv: https://arxiv.org/abs/1906.00121

### R(2+1)D
Tran, D., Wang, H., Torresani, L., Ray, J., LeCun, Y., & Paluri, M. (2018).
*A Closer Look at Spatiotemporal Convolutions for Action Recognition*.
IEEE Conference on Computer Vision and Pattern Recognition (CVPR).
arXiv: https://arxiv.org/abs/1711.11248

### Kinetics-400
Kay, W., Carreira, J., Simonyan, K., Zhang, B., Hillier, C., Vijayanarasimhan, S.,
Viola, F., Green, T., Back, T., Natsev, P., Suleyman, M., & Zisserman, A. (2017).
*The Kinetics Human Action Video Dataset*.
arXiv: https://arxiv.org/abs/1705.06950

### Whisper

Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2023).
*Robust Speech Recognition via Large-Scale Weak Supervision*.
Proceedings of the 40th International Conference on Machine Learning (ICML).
https://arxiv.org/abs/2212.04356

Used in Week 7.6b as `openai/whisper-tiny` via HuggingFace `transformers`
for emergency-phrase keyword detection.

### Qwen 2.5

Qwen Team. (2024). *Qwen2.5 Technical Report.* arXiv preprint
arXiv:2412.15115.
https://arxiv.org/abs/2412.15115

Used in Week 8 as `Qwen/Qwen2.5-1.5B-Instruct` (4-bit NF4) via
HuggingFace `transformers` for incident report generation.
Quantized with `bitsandbytes`.

## Libraries

### PyTorch
Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., ... & Chintala, S. (2019).
*PyTorch: An Imperative Style, High-Performance Deep Learning Library*.
Advances in Neural Information Processing Systems (NeurIPS).
arXiv: https://arxiv.org/abs/1912.01703

### HuggingFace Transformers
Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., ... & Rush, A. M. (2020).
*Transformers: State-of-the-Art Natural Language Processing*.
Conference on Empirical Methods in Natural Language Processing (EMNLP).
arXiv: https://arxiv.org/abs/1910.03771

### scikit-learn
Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, E. (2011).
*Scikit-learn: Machine Learning in Python*.
Journal of Machine Learning Research, 12, 2825–2830.
https://scikit-learn.org/

### LightGBM
Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., ... & Liu, T. Y. (2017).
*LightGBM: A Highly Efficient Gradient Boosting Decision Tree*.
Advances in Neural Information Processing Systems (NeurIPS).
https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree

### XGBoost
Chen, T., & Guestrin, C. (2016).
*XGBoost: A Scalable Tree Boosting System*.
ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.
arXiv: https://arxiv.org/abs/1603.02754

### Optuna
Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019).
*Optuna: A Next-generation Hyperparameter Optimization Framework*.
ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.
arXiv: https://arxiv.org/abs/1907.10902

### MLflow
Databricks. *MLflow: An Open Source Platform for the Machine Learning Lifecycle*.
https://mlflow.org/
Paper: Zaharia, M., Chen, A., Davidson, A., Ghodsi, A., Hong, S. A., Konwinski, A., ... & Zumar, C. (2018).
*Accelerating the Machine Learning Lifecycle with MLflow*.
IEEE Data Engineering Bulletin, 41(4), 39-45.

### decord
*Decord: An Efficient Video Reader for Deep Learning*.
https://github.com/dmlc/decord

## Tools

### ultralytics
Jocher, G., Chaurasia, A., & Qiu, J. (2023). *Ultralytics* (Version 8.0.0) [Computer software].
https://github.com/ultralytics/ultralytics
License: AGPL-3.0.

## Notes

- All datasets used are publicly available and used under their original licenses.
- All models are pretrained or fine-tuned using the papers and libraries above.
- This file is maintained as new datasets/models are added.