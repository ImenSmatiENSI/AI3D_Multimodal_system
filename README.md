# AI3D
AI3D: Contour Abstraction for Deep Neural Network's Certification against 3D Displacement

Traditional deep-learning verification tools often deal with image-type data. Such systems certify the robustness of the image classifier against attacks including simple contrast, FGSM noise, and $L_{\infty}$. Very few studies deal with geometric certification which is still a remaining challenge. On contour-type data, the robustness of a given NN-based classifier is a critical issue. To assess it. In this work, we propose a general framework of Lower and Upper Bounds that enables us to verify the robustness of a DNN against a broader range of projective attacks represented including 3D rotation and 3D translation.
We implement it as a system which is the first that certifies deep contour classifiers. Additionally, we integrated this into a multimodal system that verifies images based on both pixel representations and contours. We propose to name this system AI3D. We tested it on two different image datasets, extracting their contours using an affine arclength reparametrization approach to represent contours with (X, Y) coordinates.

## Overview
This repository presents an adaptation of the DeepPoly verifier within the ERAN system, transforming it into a certification framework based on abstract interpretation for multimodal datasets (images and contours). Our approach extends DeepPoly to handle more complex adversarial attacks, such as projective attacks, by modifying the main verification framework initially implemented in ERAN.
The AI3D framework is implemented in python programming language and supports any architecture as fully connected, convolutional, and max pooling layers for contours' classification. 
Based on the DeepPoly analyzer, a domain that combines Polyhedrons with Intervals. It is implemented using two main libraries: [ERAN](https://github.com/eth-sri/eran) and [ELINA](https://github.com/eth-sri/ELINA) library, coded in respectively Python and C programming languages.


![This figure](SL2_butterfly.png)  This figure illustrates the concept of the system, depicting various perturbations within the projective transformation, such as 3D translation or 3D rotation. 



# Reproducing the Results

If your aim is to implement neural network verification in your application, it's advisable to employ the latest [ERAN](https://github.com/eth-sri/eran) verifier, which boasts enhanced verification algorithms rooted in abstract interpretation. To incorporate this verifier and enable contour support for any object tested with deepPoly, consult the adapted main framework. When selecting the abstract domain, consider its suitability for representing the specific attacks tested in this research.

## 🚀 Installation Instructions for AI3D_Multimodal_System
If your aim is to implement neural network verification in your application, it's advisable to employ the latest [ERAN](https://github.com/eth-sri/eran) verifier, which boasts enhanced verification algorithms rooted in abstract interpretation. To incorporate this verifier and enable contour support for any object tested with deepPoly, consult the adapted main framework. When selecting the abstract domain, consider its suitability for representing the specific attacks tested in this research.
To use the AI3D_Multimodal_System, you first need to set up ERAN (as it integrates the DeepPoly verifier). Follow these steps to install ERAN, then proceed to install the rest of the dependencies for AI3D_Multimodal_System.

### Clone the ERAN Repository

First, clone the official ERAN repository to your local machine:
```bash
git clone https://github.com/eth-sri/eran.git
cd eran
```
### Set Up ERAN
Follow the setup instructions in the ERAN GitHub repository to configure ERAN. This setup will enable the framework for robustness verification.

### Clone the AI3D_Multimodal_System Repository 
After setting up ERAN, we have updated the main framework initially provided by ERAN to support multimodal data (images and contours). ERAN itself only supports some basic image examples, but our system allows for testing on more complex datasets, including both images and contours. This extension is necessary to make DeepPoly work effectively on multimodal data.
   ```bash
   git clone https://github.com/ImenSmatiENSI/AI3D_Multimodal_System.git
   cd AI3D_Multimodal_System
```
### Install dependencies:
   ```bash
   pip install -r requirements.txt
```

## Dataset Preparation
All steps for embedding contours and preprocessing image data, including contour extraction, are detailed in the "Multimodal Extraction" repository on my GitHub. You can find the repository [Contour extraction](https://github.com/ImenSmatiENSI/Multimodal_extraction).
This project extends the experiments to include multimodal datasets such as the Swedish Leaf dataset.

### Application:Swedish Leaf Dataset & Preprocessing
We applied our approach to a real-world dataset: Swedish Leaf, demonstrating its applicability to practical scenarios.
We introduced a dedicated folder ([Application_SwedishLeaf](https://github.com/ImenSmatiENSI/AI3D_Multimodal_system/tree/main/SwedishLeaf_Application)) containing:
📌 Dataset Preprocessing: Scripts to extract contours and preprocess images for certification.
📌 Contour Extraction: Converts images into edge-based representations for multimodal analysis.
📌 Projective Transformation Study: The dataset is adapted to include projective distortions, and a corresponding image is provided to illustrate these transformations.

To preprocess the Swedish Leaf dataset and extract contours Open the Preprocessing Notebook
In Google Colab, open the notebook preprocess_swedish_leaf.ipynb. 
If you have the notebook in your repository, you can open it directly in Colab by navigating to the file in your Google Drive or GitHub.
run:
 ```bash
https://colab.research.google.com/github/ImenSmatiENSI/AI3D_Multimodal_system/Application_SwedishLeaf/preprocess_swedish_leaf.ipynb
```

## Reproducing the Results
After completing all the installation steps and adapting the code (including modifications to the main framework and contour extraction), you can proceed to verify the system. To run the verification, simply execute the following command, ensuring that you specify the path to the contours dataset:
 
 ```bash
~/ERAN/tf_verify$ python3 . --netname models_Leaf/leaf_contour_model3NN.pb --domain deeppoly --dataset leafcontours
```
