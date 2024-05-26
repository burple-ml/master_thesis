# CNNs to diagnose Diabetes Mellitus 2 using Raman Spectroscopy of the skin

Thesis Project titled - "**Convolutional Neural Networks for Raman Spectrum Classification**"

# **Summary**
* built shallow CNNs to classify raman spectra of Diabetic(DM2) patients, and healthy controls. The performance of the network
with and without preprocessing (Baseline correction and PCA) is compared. 
* K-Nearest Neighbors, a popular classifier for raman spectra, and one that is used in many raman softwares, is built as a baseline model. Its performance is also measured with and without preprocessing. 
* Hyperparameter search was performed for both classifiers to choose the best hyperparameters.
* The performance of both these models is measured with a DEEP Convolutional Neural Network architecture that was proposed by 
[Liu et. al](https://arxiv.org/abs/1708.09022) as a unified solution, and one that classifies raw raman spectra. Thus the unified solution of Liu et. al is considered a true end-to-end solution for raman spectrum classification. 
* The Deep Neural Network was trained using the [RRUFF minerals dataset](https://rruff.info/zipped_data_files/), and the shallow CNNs were trained on the [kaggle dataset of Raman spectra of the skin](https://www.kaggle.com/codina/raman-spectroscopy-of-diabetes/version/8)
* Transfer learning was performed on the Deep CNN, trained on minerals dataset, to classify raman spectra of the skin. 
* Several experiments with simulated data using a voigt data generator were also conducted. 
* Student-t test was used to establish statistical significance in the comparison of the performance of the different models.

# **ORIGINAL DATASET AND PREPROCESSING:-**
* As mentioned earlier, the spectral dataset of the skin can be found on the [kaggle page here](https://www.kaggle.com/codina/raman-spectroscopy-of-diabetes/version/8). I have included the dataset under the `/dataset` folder.
* A preprocessing kernel written in matlab by the authors of the [original study, Guevara et. al](https://www.osapublishing.org/boe/abstract.cfm?uri=boe-9-10-4998) , is used to perform baseline correction. It is an implementation of the vancouver raman algorithm. Their implementation can be found [here](https://github.com/guevaracodina/raman-diabetes/) . This kernel was used to process the raw spectral dataset, and the baseline corrected dataset is the .mat file called - `BaselineCorrectedData2.mat`
* The Voigt Data Generator module used by me is written by Tommy Sonne Alstrøm and can be found [here](https://lab.compute.dtu.dk/tsal/A-pseudo-Voigt-component-model-for-high/). Thus the simulated data is generated with the help of the `psuedovoigt.m` matlab kernel found in this project. The generated data is included as .mat files under `dm2_diagnosis/experiments_simulateddata/` . The matlab code for generation of these .mat files along with the `pseudovoigt.m` module can be found under `voigt_datagenerator/` in the same path.
* The Dataset used to train the Deep Convolutional Neural Network is the RRUFF mineral dataset, and is found [here](https://rruff.info/zipped_data_files/). I have used the low resolution raman dataset (titled `LR-Raman`). It is included under a folder with the same name

# **CODE FOR THE EXPERIMENTS:**
* The experiments with the baseline corrected data, with raw data, with Deep CNN, and simulated data, can be found under their respective folders. 
* Each folder contains IPython Notebooks with the analysis and implementation of the different models. The Checkpoints are also included, so you should be able to see the results when you clone the repository. 
* The hyperparameter search and training of the Deep CNN are computationally demanding tasks, and hence they were carried out on [DTU HPC Clusters](https://www.hpc.dtu.dk/) . The scripts for these are also included, and can be found under the  `hyperparameter_search/` folder and `experiments_deepcnn/deepcnn_training/` folder respectively. 
* The trained Deep CNN model is saved under the `DEEP_CNN_TRAINEDMODEL/` folder.
* Several plots and variables were recorded during these experiments, and hence the appearance of the `ImportantPlots/` , `ImportantVariables/` and the occasional `Images/` folder. These images and plots are all included in the thesis report. 
* The scripts titled `reading_raman.py` are used to read the dataset and pack them into neat numpy arrays.