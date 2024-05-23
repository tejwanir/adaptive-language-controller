import json
import numpy as np
import pathlib
import tqdm
import torch
import torch.utils.data as tud
import transformers
from transformers import BertTokenizerFast
import matplotlib.pyplot as plt

import data_processing_phrase as dpp

# from https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
# for json dumping np arrays

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# adapted from KNN branch: https://github.com/tejwanir/adaptive-language-controller/blob/KNN/knn.py
def parse_transcription_to_phrases(
    transcription, use_transcription_segments: bool = False
):
    import re

    import spacy
    from deepmultilingualpunctuation import PunctuationModel

    nlp = spacy.load("en_core_web_trf")
    punct_model = PunctuationModel()

    segments = []
    # Options 1: Use the segments from the transcription
    all_segments = transcription["segments"]
    if not use_transcription_segments:
        # Options 2: Merge all segments into a single segment
        mega_segment = {
            "text": " ".join([s["text"].strip() for s in all_segments]),
            "words": sum((s["words"] for s in all_segments), []),
        }
        all_segments = [mega_segment]

    for segment in all_segments:
        text: str = segment["text"].strip()
        text_fixed = punct_model.restore_punctuation(text)
        phrases = [p for p in re.split(r"[.!?,]", text_fixed) if p]
        next_word_idx = 0
        normalize = lambda s: re.sub(r"\s+", " ", s.strip().lower()).replace("' ", "'")
        for phrase in phrases:
            phrase = normalize(phrase)
            phrase_words = phrase.split()
            phrase_idx = 0
            start_idx = next_word_idx
            while (
                next_word_idx < len(segment["words"])
                and phrase_idx < len(phrase_words)
                and normalize(
                    re.sub(r"[.!?,]", "", segment["words"][next_word_idx]["word"])
                )
                == phrase_words[phrase_idx]
            ):
                next_word_idx += 1
                phrase_idx += 1
            start_time = segment["words"][start_idx]["start"]
            end_time = segment["words"][next_word_idx - 1]["end"]
            segments.append(
                {
                    "phrase": phrase,
                    "start_time": start_time,
                    "end_time": end_time,
                }
            )

    curr_filepath = pathlib.Path(__file__).parent.resolve()
    with open(f"{curr_filepath}/phrases.json", "w") as f:
        json.dump(segments, f, indent=2)
    return segments

def eval_visualization(eval_json):
    # create histograms of metrics

    with open(eval_json, 'r') as f:
        metrics = json.load(f)

    print(len(metrics))
    bleu_scores, rouge_scores = [], []
    bert_scores_precision, bert_scores_recall, bert_scores_f1 = [], [], []
    for ix, data in metrics.items():
        bleu_scores.append(data['bleu_score']['bleu'])
        rouge_scores.append(data['rouge_score']['rouge1'])
        bert_scores_precision.append(data['bert_score']['precision'])
        bert_scores_recall.append(data['bert_score']['recall'])
        bert_scores_f1.append(data['bert_score']['f1'])

    plt.hist(bleu_scores, bins=50)
    plt.title('BLEU scores')
    plt.show()

    plt.hist(rouge_scores, bins=50)
    plt.title('ROUGE scores')
    plt.show()

    # plt.hist(bert_scores_precision, bins=50)
    # plt.show()

def misc_metrics():
    file_range = range(1,7)
    phrases_fp_list, pose_fp_list, force_fp_list = [], [] ,[]
    for i in file_range:
        phrases_fp_list.append(f'data/lightbuzz_table_{i}/phrases.json')
        pose_fp_list.append(f'data/lightbuzz_table_{i}/cut_poses.jsonl')
        force_fp_list.append(f'data/lightbuzz_table_{i}/cut_data.csv')

    tokenizer = transformers.BertTokenizerFast.from_pretrained('bert-base-uncased')

    words_set = set()
    phrase_len_total = 0
    phrase_len_count = 0
    for phrases_fp, pose_fp, force_fp in zip(phrases_fp_list, pose_fp_list, force_fp_list):
        text_df, phrases_json = dpp.preprocess_text_data(phrases_fp, tokenizer)

        for d in phrases_json:
            phrase = d['phrase']
            new_words_set = set(phrase.split())
            words_set |= new_words_set

            phrase_len_total += len(phrase.split())
            phrase_len_count += 1

    print(f'vocab size: {len(words_set)}')
    print(f'avg phrase seq length: {phrase_len_total / phrase_len_count}')


if __name__ == '__main__':
    # eval_visualization('eval.json')

    misc_metrics()