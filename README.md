# TADA: Training-free Attribution and Out-of-Domain Detection of Audio Deepfakes

This is the official repository for the Interspeech 2025 submission:
> Adriana Stan, David Combei, Dan Oneata, Horia Cucu, "TADA: Training-free Attribution and Out-of-Domain Detection of Audio Deepfake", accepted at Interspeech 2025, Rotterdam, Netherlands.

Citation

```bibtex
@inproceedings{stan_interspeech25,
               author={Stan, Adriana and Combei, David and Oneata, Dan and Cucu, Horia},
               booktitle={Proc. of Interspeech}, 
               title={{TADA: Training-free Attribution and Out-of-Domain Detection of Audio Deepfake}}, 
               year={2025}}
```




## Setup

1) Datasets
   
   Download the datasets from the original repositories:
   - [ASV19](https://datashare.ed.ac.uk/handle/10283/3336)
   - [ASV21](https://www.asvspoof.org/index2021.html)
   - [ASV5](https://zenodo.org/records/14498691)
   - [MLAAD v5](https://deepfake-total.com/mlaad)
   - [TIMIT](https://zenodo.org/records/6560159)
  
  You can then filter out the files according to the protocols available in the `dataset_protocols/` folder.

2) Pretrained SSL Model and Feature Extraction
   
   In our implementation we used the [w2v-bert-2.0](https://huggingface.co/facebook/w2v-bert-2.0) pretrained, frozen model.

   You can extract the average pooled representations of the audio files using the
    ```
   python wav2vec_bert_extractor.py
    ```

   script. The output will consist of a single `.npy` file for each dataset containing the features in the order listed in the protocols file. 


## Running the individual experiments from the paper

### 1) Checkpoint attribution
   
   For the simple checkpoint attribution based on layer 4 features of the w2v-bert-2.0 model, you can use the

   ```
   run_checkpoint_attribution.py
   ```

   script. The script will use the complete set of features (from all datasets) as listed in `config.py`, and split them into train and test partitions at a 80:20 ratio. It will then fit a kNN with k=21 neighbours and display the classification report. You can optionally use a standard scaler before fitting the kNN (just uncomment that part of the code).

   If you also opt to print or save the confusion matrix, it should resemble Figure 1 from the paper, with very few mis-attributions across the datasets. The full matrix is also available [here](confusion_21_neighbors.pdf). The results are also saved to `results_checkpoint_attribution.log`.
   
### 2) Speaker classification
   
   For the LJSpeech systems' attribution as listed at the end of Section 3.2 you can use the 
   
   ```
   python run_ljspeech_classification.py
   ``` 
   
   script. The result (with some degree of randomness due to the train-test split if the seed is not set) should look something like:
    
    ```
                                                  precision    recall  f1-score   support
                                      LJSpeech       0.89      0.96      0.92       419
          en_tts_models_en_ljspeech_fast_pitch       0.78      0.89      0.83        87
            en_tts_models_en_ljspeech_glow-tts       0.93      0.97      0.95       104
          en_tts_models_en_ljspeech_neural_hmm       0.96      0.85      0.90       100
            en_tts_models_en_ljspeech_overflow       0.87      0.88      0.88       111
       en_tts_models_en_ljspeech_speedy-speech       0.84      0.91      0.88       105
       en_tts_models_en_ljspeech_tacotron2-DCA       1.00      0.89      0.94       103
       en_tts_models_en_ljspeech_tacotron2-DDC       0.94      0.74      0.83       103
    en_tts_models_en_ljspeech_tacotron2-DDC_ph       0.94      0.88      0.91       102
                en_tts_models_en_ljspeech_vits       0.50      0.53      0.51        98
          en_tts_models_en_ljspeech_vits--neon       0.49      0.41      0.45        98
                                  accuracy                           0.84      1430
                                 macro avg       0.83      0.81      0.82      1430
                              weighted avg       0.85      0.84      0.84      1430
    ```


Similarly, for the multispeaker checkpoints, you can use 

```
python3 run_speaker_classif_multispeaker_systems.py
``` 

Results from these steps are saved to `results_ljspeech_classification.log` and `results_multispeaker_systems_classification.log`.

### 3) Out of distribution (OOD) detection

   The last section of the paper introduced the OOD detection of novel checkpoints based on the kNN distance. You can run the OOD detection using: 
   
   ```
   python run_ood.py
   ``` 
   The output will show the results for each of the 5 datasets in terms of OOD detection accuracy. 5 checkpoints from each dataset are set aside: 2 for validation and 2 for testing.



# Acknowledgements
This work was co-funded by EU Horizon project AI4TRUST (No. 101070190), and by the Romanian Ministry of Research, Innovation and Digitization project DLT-AI SECSPP (ID: PN-IV-P6-6.3-SOL-2024-2-0312).
