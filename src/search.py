import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from pydub import AudioSegment
from transformers import (
    AutoFeatureExtractor,
    BertForSequenceClassification,
    BertJapaneseTokenizer,
    Wav2Vec2ForXVector,
)


class Search:
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
        self.text_reference = torch.load(self.config.path_text_embedding)
        self.audio_reference = torch.load(self.config.path_audio_embedding)
        self.similarity = torch.nn.CosineSimilarity(dim=-1)

    def search(self, text, audio, ratio, topk):
        text_embed, audio_embed = self.get_embedding(text, audio)
        if text_embed is not None and audio_embed is not None:
            result = self.similarity(
                text_embed, self.text_reference
            ) * ratio + self.similarity(audio_embed, self.audio_reference) * (1 - ratio)
        elif text_embed is not None:
            result = self.similarity(text_embed, self.text_reference)
        elif audio_embed is not None:
            result = self.similarity(audio_embed, self.audio_reference)
        else:
            raise ValueError("Input text or upload audio file.")

        rank = np.argsort(result.numpy())[::-1][0:topk]
        return self.df.iloc[rank]

    def get_embedding(self, text, audio):
        text_embed = None if text == "" else self._get_text_embedding(text)
        audio_embed = None if audio == "" else self._get_audio_embedding(audio)
        return text_embed, audio_embed

    def _get_text_embedding(self, text):
        tokenized_text = self.text_tokenizer.tokenize(text)
        indexed_tokens = self.text_tokenizer.convert_tokens_to_ids(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens])
        with torch.no_grad():
            all_encoder_layers = self.text_model(tokens_tensor)
        embedding = torch.mean(all_encoder_layers[1][-2][0], axis=0).reshape(1, -1)
        return embedding

    def _get_audio_embedding(self, audio):
        song = AudioSegment.from_wav(audio)
        song = np.array(song.get_array_of_samples(), dtype="float")
        inputs = self.audio_feature_extractor(
            [song],
            sampling_rate=self.config.sample_rate,
            return_tensors="pt",
            padding=True,
        )
        with torch.no_grad():
            embedding = self.audio_model(**inputs).embeddings
        return embedding
