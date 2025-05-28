# TADA: Training-free Attribution and Out-of-Domain Detection of Audio Deepfakes

This is the official repository for the Interspeech 2025 submission:
> Adriana Stan, David Combei, Dan Oneata, Horia Cucu, "TADA: Training-free Attribution and Out-of-Domain Detection of Audio Deepfake", accepted at Interspeech 2025, Rotterdam, Netherlands.

Citation

```
@inproceedings{stan_interspeech25,
               author={Stan, Adriana and Combei, David and Oneata, Dan and Cucu, Horia},
               booktitle={Proc. of Interspeech}, 
               title={{TADA: Training-free Attribution and Out-of-Domain Detection of Audio Deepfake}}, 
               year={2025}}
```


--

## Setup

1) Datasets
   Download the datasets from the original repositories:
   - [ASV19](https://datashare.ed.ac.uk/handle/10283/3336)
   - [ASV21](https://www.asvspoof.org/index2021.html)
   - [ASV5](https://www.asvspoof.org/workshop2024)
   - [MLAAD v5](https://deepfake-total.com/mlaad)
   - [TIMIT](https://zenodo.org/records/6560159)
  
  You can then filter out the files according to the protocols available in the `dataset_protocols/` folder.

2) Pretrained SSL Model and Feature Extraction
   In our implementation we used the [w2v-bert-2.0](https://huggingface.co/facebook/w2v-bert-2.0) pretrained, frozen model.
   You can extract the average pooled representations of the audio files using the `wav2vec_bert_extractor.py` script. The output will consist of a single `.npy` file containing the features in the order listed in the protocol. 


## Running the individual experiments

1) Model attribution
   For the simple model attribution based on the layer 4 features of the w2v-bert-2.0 model, you can use the `run_model_attribution.py` script. The script will use the complete set of features as listed in `config.py`, and split them into train and test partitions at a 80:20 ratio. It then fits a kNN with k=21 and displays the classification report. You can optionally use a standard scaler before fitting the kNN (just uncomment that part of the code).

   If you also opt to print or save the confusion matrix, it should resemble Figure 1 from the paper, with very few confusions across the datasets. 
   
2) Speaker classification
   For the LJSpeech systems' attribution as listed at the end of Section 3.2 you can use the `.py` script
   Similarly, for the multispeaker checkpoints, you can use the `.py` script. 

 
3) OOD detection
   The last section of the paper introduced the OOD detection of novel checkpoints based on the kNN distance. The first step to achieve this is to compute...



