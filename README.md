Cross-Individual Affective Detection Using EEG Signals with Audio-Visual Embedding.
=
* A Pytorch implementation of our paper "Cross-Individual Affective Detection Using EEG Signals with Audio-Visual Embedding.".<br> 
* [arxiv](https://arxiv.org/abs/2202.06509)
# Installation:
* Python 3.7
* Pytorch 1.3.1
* NVIDIA CUDA 9.2
* Numpy 1.20.3
* Scikit-learn 0.23.2
* scipy 1.3.1
# Preliminaries
* Prepare dataset: [DEAP](http://www.eecs.qmul.ac.uk/mmv/datasets/deap/) and [MAHNOB-HCI](https://mahnob-db.eu/hci-tagging/)
# Training 
* Msdann model definition file: model_mutiscale.py 
* Pipeline of the Msdann: train_and_test_deap_Msdann_de.py,train_and_test_hci_Msdann_de.py
* implementation of domain adversarial training: Adversarial.py
# Usage
* After modify setting (path, etc), just run the main function in the train_and_test_deap_Msdann_de.py(DEAP dataset) or train_and_test_hci_Msdann_de.py(HCI dataset)
# Acknowledgement
* The implementation code of domain adversarial training is bulit on the [dalib](https://dalib.readthedocs.io/en/latest/index.html) code base 
# Citation
If you find our work helps your research, please kindly consider citing our paper in your publications.
@article{liang2022cross,
  title={Cross-individual affective detection using EEG signals with audio-visual embedding},
  author={Liang, Zhen and Zhang, Xihao and Zhou, Rushuang and Zhang, Li and Li, Linling and Huang, Gan and Zhang, Zhiguo},
  journal={Neurocomputing},
  volume={510},
  pages={107--121},
  year={2022},
  publisher={Elsevier}
}
