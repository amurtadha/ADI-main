 
 # Arabic Dialect Identification 

 This is the source code for the paper: Mohammed Abdelmajeed, Ahmed Murtadha et al. "Leveraging Unlabeled Corpus for Arabic Dialect Identification". 
 
 We aim to leverage unlabeled data to enhance the performance of deep neural networks on Arabic Dialect Identification in adversarial settings.
 

# Data



The unlabeled data used in our experminents can be downloaded from this [link]([https://drive.google.com/file/d/1NYm5CVXK7vqn-Nf8rnin-4iAxWeJcKVv/view?usp=sharing](https://drive.google.com/file/d/1qJImRVG-q8hjrSIk7VkcOIv-83Yhm3_v/view?usp=sharing)). 


# Prerequisites:
Required packages are listed in the requirements.txt file:

```
pip install -r requirements.txt
```
Please update the workspace path

# To train the model
*  Run cd scripts/:
  ```
  sh train_ours.sh
  ```
  
*  Or directly use:
```
CUDA_VISIBLE_DEVICES=0 python train.py --dataset Nadi --train_sample 0.1 --pretrained_bert_name /workspace/plm/arbert
```

- The params :
    - --dataset =\{Nadi\}
    - --train_sample  # the percentage of labeled training samples
    - --pretrained_bert_name # the path of the pretrained language model

The models are saved to models
The results are written to outputs

If you use the code,  please cite the paper: 
 ```
```
