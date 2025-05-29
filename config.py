meta_dir = './sorted_protocols/'
metadata = [
          "asv19_train_spk_model.txt",
          "asv19_dev_spk_model.txt",
          "asv19_eval_spk_model.txt",
          "asv21_eval_fakes_nocodecs_noasv19_spk_model.txt",
          "asv5_train_spk_model.txt",                                   
          "asv5_dev_spk_model.txt",
          "asv5_eval_nocodecs_spk_model.txt",
          "timit_clean_spk_model.txt",
          "mlaad_v5_spk_model.txt",

]

feats_dir = "./features/wav2vec-bert/"

feats = [
        f"wav2vec2-bert_Layer4_asv19_train.npy",
        f"wav2vec2-bert_Layer4_asv19_dev.npy",
        f"wav2vec2-bert_Layer4_asv19_eval.npy",
        f"wav2vec2-bert_Layer4_asv21.npy",
        f"wav2vec2-bert_Layer4_asv5_train.npy",
        f"wav2vec2-bert_Layer4_asv5_dev.npy",
        f"wav2vec2-bert_Layer4_asv5_eval_nocodec.npy",
        f"wav2vec2-bert_Layer4_timit_tts_clean_new.npy",
        f"wav2vec2-bert_Layer4_mlaad_v5.npy",
    ]





