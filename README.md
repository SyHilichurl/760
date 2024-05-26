# 760

## How to use the Code

v3.py is the InceptionV3 model from website.(website in Reference)


baseline.py  It is the baseline model for just InceptionV3.We get a base InceptionV3 model from website.And change it to fit our experiment.


with_cutmix_and_se()use CV).py is the final model fro our study.




The remaining Python files incrementally add various techniques to the baseline model until reaching the final model.（With techniques as it's name）




Each model code includes the complete process of reading data, building the model, training the model, and evaluating the model.



test for new one picture.py is you can ues this code to test the model on new picture,



## Reference
GoogLeNet(InceptionV3) website : https://blog.csdn.net/yangyu0515/article/details/134371005

Data Source website: https://www.kaggle.com/datasets/anshtanwar/pets-facial-expression-dataset


Li, J., Jin, K., Zhou, D., Kubota, N., & Ju, Z. (2020). Attention mechanism-based CNN for facial expression recognition. Neurocomputing, 411(S0925-2312(20)30983-8), 340–350. https://doi.org/10.1016/j.neucom.2020.06.014


Mao, Y., & Liu, Y. (2023). Pet dog facial expression recognition based on convolutional neural network and improved whale optimization algorithm. Scientific Reports, 13(1). https://doi.org/10.1038/s41598-023-30442-0


Szegedy, C., Vanhoucke, V., Ioffe, S., & Shlens, J. (2015). Rethinking the Inception Architecture for Computer Vision.

CutMix:https://blog.csdn.net/qq_44949041/article/details/129590645
