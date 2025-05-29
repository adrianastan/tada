import os
import sys
from tqdm import tqdm
import numpy as np
import torch
import librosa
from transformers import AutoFeatureExtractor, Wav2Vec2Model, Wav2Vec2BertConfig, Wav2Vec2BertModel


class HuggingFaceFeatureExtractor:
    def __init__(self, model_class, name=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(name)
        self.model = model_class.from_pretrained(name, output_hidden_states=True)
        self.model.eval()
        self.model.to(self.device)

    def __call__(self, audio, sr):
        inputs = self.feature_extractor(
            audio,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.hidden_states


FEATURE_EXTRACTOR = {
    "wav2vec2-bert": lambda: HuggingFaceFeatureExtractor(
        Wav2Vec2BertModel, "facebook/w2v-bert-2.0"
    ),
}


def read_metadata(file_path):
    relevant_files = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) > 1:
                relevant_files.append(parts[0])
                
    ## Pay attention to this list!!!!
    return relevant_files



def main(outdir,indir, metadata_file, dataset):
    ## Get a list of audio files
    relevant_files = read_metadata(metadata_file)
    
    print(f"Metadata contains {len(relevant_files)} files.")
    model_name = 'wav2vec2-bert'
    feature_extractor = FEATURE_EXTRACTOR[model_name]()

    layer_embeddings = []
    ## Keep track of problematic files
    fout = open(os.path.join(outdir,"bad_data.txt"), 'w')
    
    ## Create the output folder
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        
    for fi in tqdm(relevant_files[:]):
            fi = f'{os.path.join(indir,fi)}'
            try:
                audio, sr = librosa.load(fi, sr=16000)
                hidden_states = feature_extractor(audio, sr)

                layer_output = hidden_states[4] #4th tsf
                mean_layer_output = torch.mean(layer_output, dim=1).cpu().numpy()
                layer_embeddings.append(mean_layer_output)

            except:
                print(f"ERROR when extracting {fi}")
                fout.write(fi+'\n')
                fout.flush()

    stacked_embeddings = np.vstack(layer_embeddings)
    ## Save the features in a single numpy file
    np.save(os.path.join(outdir, f'{model_name}_Layer5_{dataset}.npy'), stacked_embeddings)


if __name__ == '__main__':
    ## PATHS
    indir = "./audio_deepfake_datasets/TIMIT-TTS/CLEAN"
    outdir = "./features/"
    metadata_file = "./dataset_protocols/timit_clean_spk_model.txt"
    dataset = "timit_clean"
    
    main(outdir, indir, metadata_file, dataset)
    print("Done!")
