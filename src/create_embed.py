import argparse
import os

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from pydub import AudioSegment
from tqdm import trange
from transformers import (
    AutoFeatureExtractor,
    BertForSequenceClassification,
    BertJapaneseTokenizer,
    Wav2Vec2ForXVector,
)


class Embeder:
    def __init__(self, config):
        self.config = OmegaConf.load(config)
        self.df = pd.read_csv(config.path_csv)
        self.audio_feature_extractor = AutoFeatureExtractor.from_pretrained(
            "anton-l/wav2vec2-base-superb-sv"
        )
        self.audio_model = Wav2Vec2ForXVector.from_pretrained(
            "anton-l/wav2vec2-base-superb-sv"
        )
        self.text_tokenizer = BertJapaneseTokenizer.from_pretrained(
            "cl-tohoku/bert-base-japanese-whole-word-masking"
        )
        self.text_model = BertForSequenceClassification.from_pretrained(
            "cl-tohoku/bert-base-japanese-whole-word-masking",
            num_labels=2,
            output_attentions=False,
            output_hidden_states=True,
        ).eval()

    def run(self):
        self._create_audio_embed()
        self._create_text_embed()

    def _create_audio_embed(self):
        audio_embed = None
        idx = []
        for i in trange(len(self.df)):
            audio = []
            song = AudioSegment.from_wav(
                os.path.join(
                    self.config.path_data,
                    "new_" + self.df.iloc[i]["filename"].replace(".mp3", ".wav"),
                )
            )
            song = np.array(song.get_array_of_samples(), dtype="float")
            audio.append(song)
            inputs = self.audio_feature_extractor(
                audio,
                sampling_rate=self.config.sample_rate,
                return_tensors="pt",
                padding=True,
            )
            try:
                with torch.no_grad():
                    embeddings = self.audio_model(**inputs).embeddings
                audio_embed = (
                    embeddings
                    if audio_embed is None
                    else torch.concatenate([audio_embed, embeddings])
                )
            except Exception:
                idx.append(i)

        audio_embed = torch.nn.functional.normalize(audio_embed, dim=-1).cpu()
        self.clean_and_save_data(audio_embed, idx)
        self.df = self.df.drop(index=idx)
        self.df.to_csv(self.config.path_csv, index=False)

    def _create_text_embed(self):
        text_embed = None
        for i in range(len(self.df)):
            sentence = self.df.iloc[i]["filename"].replace(".mp3", "")
            tokenized_text = self.text_tokenizer.tokenize(sentence)
            indexed_tokens = self.text_tokenizer.convert_tokens_to_ids(tokenized_text)
            tokens_tensor = torch.tensor([indexed_tokens])
            with torch.no_grad():
                all_encoder_layers = self.text_model(tokens_tensor)
            embedding = torch.mean(all_encoder_layers[1][-2][0], axis=0).reshape(1, -1)
            text_embed = (
                embedding
                if text_embed is None
                else torch.concatenate([text_embed, embedding])
            )
        text_embed = torch.nn.functional.normalize(text_embed, dim=-1).cpu()
        torch.save(text_embed, self.config.path_text_embedding)

    def clean_and_save_data(self, audio_embed, idx):
        clean_embed = None
        for i in range(1, len(audio_embed)):
            if i in idx:
                continue
            else:
                clean_embed = (
                    audio_embed[i].reshape(1, -1)
                    if clean_embed is None
                    else torch.concatenate([clean_embed, audio_embed[i].reshape(1, -1)])
                )
        torch.save(clean_embed, self.config.path_audio_embedding)


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config.yaml",
        help="File path for config file.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argparser()
    embeder = Embeder(args.config)
    embeder.run()
