 
 # Arabic Dialect Identification 

 This is the source code for the paper: Mohammed Abdelmajeed, Ahmed Murtadha et al. "Leveraging Unlabeled Corpus for Arabic Dialect Identification". 
 
 We aim to leverage unlabeled data to enhance the performance of deep neural networks on Arabic Dialect Identification in adversarial settings.
 

# Data



The unlabeled data used in our experminents can be downloaded from this [link](https://drive.google.com/file/d/1NYm5CVXK7vqn-Nf8rnin-4iAxWeJcKVv/view?usp=sharing). 


# Prerequisites:
Required packages are listed in the requirements.txt file:

```
pip install -r requirements.txt
```
# Training
*  First update the workspace path
*  Run cd scripts/:
     ```
     sh train_ours.sh
     ```
CUDA_VISIBLE_DEVICES=0 python train.py --dataset Nadi --train_sample 0.1
```     
*  Run the following code to train ADI:
```
CUDA_VISIBLE_DEVICES=0 python train.py --dataset Nadi --train_sample 0.1
```

- The params could be :
    - --dataset =\{Nadi\}
    - --train_sample ={0.1, 0.2,...,1.0},

The results are written to outputs



If you use the code,  please cite the paper: 
 ```
```
