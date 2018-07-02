# PNDC
**Pulmonary Nodule Detection & Classification**

Lung Cancer Diagnosis Algorithm in Patients with Isolated Pulmonary Nodules in Computed Tomography image of the Chest Using Deep Learning

\ | Details
 :-: | :-----:
 **Abstract** | Deep neural network to detect isolated nodules, which classify whether nodule is cancer or not from chest CT(computed tomography) images. In this project, deep neural network is based on CNN including Deconvolution layers and unpooling layers for classification and segmentation.
 Demo | <b>Original&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Predicted</b></br><img src="./images/original.png" width="256" height="256">    <img src="./images/predicted.png" width="256" height="256"> </br>Dataset is comes from **[The Cancer Imaging Archive](http://www.cancerimagingarchive.net/)**
 
 \ | [1] | [2] | [3] | [4] | [Proposal]
 :-: | :-: | :-: | :-: | :-: | :--------:
 **Accuracy** | 72% | 82.3% | 84% | 89.3% | _**92.7**_
 
## Requirements
- **Python ≥ 3.5**
- **Tensorflow ≥ 1.5.0**
- **Caffe = 1.0.0**
- **Numpy ≥ 1.14.3**
- **Tqdm ≥ 4.19.9**
- **LMDB ≥ 0.94**


#### 2 Grand Prize Awarded
<img src="./images/대상-과학기술대학장.jpeg" width="300" height="400"><img src="./images/최우수상-스마트도시과학경영대학원장.jpeg" width="300" height="400">

## Citations - for Accuracy comparison
[1] Y.Lee, T. Hara, Fujita, S.Itoh, and T. Ishihaki. "Automated detection of pulmonary nodules in helical CT images based on an improved temlate maching technique", IEEE Trans. Med. Imaging, vol.20, pp. 595-604, 2001.

[2] A. Farag, A. El-Baz, G. G. Gimelfarb, R. Falk, and R. S. Hushek, "Automatic detedction and recognition of lung abnoralities in helical CT images using deformable templates,"in Lecture Notes in Computer Science, Medical Image Computing and Computer Assisted Intervention, vol. 3217, New York: Springer-Verlag, pp. 856-864, 2004.

[3] Madero Orozco, H; VergaraVillegas, O.O. ; De jesus Ochoa Dominguez, H. ; Cruz Shanchez, V.G, "Lung Nodule Classification in CT Thorax Images Using Support VectorMachines", Artificial Intelligence (MICAI) 12th Mexican International Conference, pp. 277-283, 2013.

[4] Zihao Lua,  Marcus A. Brubaker, Michael Brudno. "Size and Texture-based Classification of Lung Tumors with 3D CNNs", 2017 IEEE Winter Conference on Applications of Computer vision, 806-814, 2017.
