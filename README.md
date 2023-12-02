# AI3DC
AI3DC: Contour Abstraction for Deep Neural Network's Certification against 3D Displacement

Traditional deep-learning verification tools often deal with image-type data. Such systems certify the robustness of the image classifier against attacks including simple contrast, FGSM noise, and $L_{\infty}$. Very few studies deal with geometric certification which is still a remaining challenge. On contour-type data, the robustness of a given NN-based classifier is a critical issue. To assess it, we propose in this paper a general formalism of Lower and Upper Bound that allows us to verify the robustness of a DNN against a larger class of geometrical attacks. We implement it as a system which is the first that certifies deep contour classifiers. We propose to call it AI3DC. Any well parameterized 2d contour dataset will be compatible with AI3DC.


This figure illustrates the concept of the system, depicting various perturbations within the projective transformation, such as 3D translation or 3D rotation. 
![figure]([[https://github.com/ImenSmatiENSI/AI3DC/main/3Dimage.png]]) 

The AI3DC framework is implemented in python programming language and supports any architecture as fully connected, convolutional, and max pooling layers for contours' classification. 
Based on the DeepPoly analyzer, a domain that combines Polyhedrons with Intervals. It is implemented using two main libraries: ERAN \footnote{Available on Github in the following link \url{https://github.com/eth-sri/eran}} and ELINA library \footnote{Available on Github at the following link \url{https://github.com/eth-sri/ELINA}}, coded in respectively Python and C programming languages.
