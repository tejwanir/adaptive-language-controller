import json
import numpy as np
import pathlib
import tqdm
import torch
import torch.utils.data as tud
import transformers
from transformers import BertTokenizerFast
import matplotlib.pyplot as plt
import evaluate
import bert_score as bert_score_lib

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

def eval_metrics(model, dataset):
    '''
    Current metrics:
    * BLEU score
      HF reference: https://huggingface.co/spaces/evaluate-metric/bleu
    * ROUGE score
      HF reference: https://huggingface.co/spaces/evaluate-metric/rouge

      Bleu measures precision: how much the words (and/or n-grams) in the machine generated summaries appeared in the human reference summaries. Rouge measures recall: how much the words (and/or n-grams) in the human reference summaries appeared in the machine generated summaries

    * BERTScore (https://arxiv.org/abs/1904.09675)
    '''
    # for now, looping over each sample in dataset individually instead of in batches
    d = {}
    bert_scorer = bert_score_lib.BERTScorer(
        lang='en',
        rescale_with_baseline=True,
    )

    for ix in tqdm.tqdm(range(len(dataset)), total=len(dataset)):
        ### load predictions and references ###
        src, references = dataset[ix]
        decoded_output = model.predict(src.unsqueeze(0), use_beam_search=False)
        # convert predictions and references to strings; remove cls/sep tokens, convert to text
        for i, token in enumerate(references):
            sep_ix = None
            if token.item() == model.tokenizer.sep_token_id:
                sep_ix = i
                break
        references = [' '.join(model.tokenizer.batch_decode(references[1:sep_ix]))]

        predictions = decoded_output.split()[1:]
        if predictions[-1] == model.tokenizer.sep_token_id:
            predictions = predictions[:-1]
        predictions = [' '.join(predictions)]

        print(f'{ix=}')
        print(f'{references=}')
        print(f'{predictions=}')


        ### BLEU ###
        bleu_scorer = evaluate.load('bleu')
        # bleu_score = dict with keys ['bleu', 'precisions', 'brevity_penalty', 'length_ratio', 'translation_length', 'reference_length']
        bleu_score = bleu_scorer.compute(predictions=predictions, references=references)
        print(f'{bleu_score=}')

        ### ROUGE ###
        rouge_scorer = evaluate.load('rouge')
        # rouge_score = dict with keys ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
        rouge_score = rouge_scorer.compute(predictions=predictions, references=references)
        print(f'{rouge_score=}')

        ### BERTScore ###
        # bert_score = tuple (precision, recall, F1)
        bert_score = bert_scorer.score(cands=predictions, refs=references)
        bert_score = {
            'precision': bert_score[0].item(),
            'recall': bert_score[1].item(),
            'f1': bert_score[2].item()
        }
        print(f'{bert_score=}')

        d[ix] = {
            'bleu_score': bleu_score,
            'rouge_score': rouge_score,
            'bert_score': bert_score,
        }

    with open('eval.json', 'w') as f:
        json.dump(d, f)

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

    plt.hist(bert_scores_precision, bins=50)
    plt.title('BERTScore Precision')
    plt.show()

    plt.hist(bert_scores_recall, bins=50)
    plt.title('BERTScore Recall')
    plt.show()

    print(f'bleu avg: {np.mean(bleu_scores)}')
    print(f'rouge avg: {np.mean(rouge_scores)}')
    print(f'bertscore_precision avg: {np.mean(bert_scores_precision)}')
    print(f'bertscore_recall avg: {np.mean(bert_scores_recall)}')

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
    eval_visualization('eval_phrase_1.3.json')

    # misc_metrics()