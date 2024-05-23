import json
import multiprocessing as mp
import numpy as np
import pandas as pd
import time
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import tqdm
import transformers
import matplotlib.pyplot as plt
import random
from copy import deepcopy
import evaluate
import bert_score as bert_score_lib

from data_processing_phrase import preprocess_pipeline
from lightbuzz_poses_transformer import collect_poses


### global constants ###
DATA_FP = 'data'
D_MODEL = 64
MODELS_FP = 'models'

class AlignedDataset(Dataset):
    def __init__(self, audio_data, pose_data, force_data, tokenizer,
                 from_records=False, from_audio_json=False):
        feature_tensor, text_tensor = preprocess_pipeline(audio_data, pose_data, force_data, tokenizer)
        self.feature_tensor = feature_tensor
        self.text_tensor = text_tensor

    def __len__(self):
        return self.feature_tensor.shape[0]

    def __getitem__(self, ix):
        return self.feature_tensor[ix], self.text_tensor[ix]

class Transformer(nn.Module):
    def __init__(self, d_model, tokenizer, max_output_length=64):
        super().__init__()

        self.d_model = d_model
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer)
        self.max_output_length = max_output_length
        self.nhead = 8
        self.num_layers = 6
        self.dim_feedforward = self.d_model

        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
        # TODO: add positional encoding
        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                self.d_model, self.nhead, self.dim_feedforward, batch_first=True
            ),
            num_layers=self.num_layers
        ).double() # TODO: need to use double? maybe use float32 for efficiency

        # TODO: generate mask
        self.decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                self.d_model, self.nhead, self.dim_feedforward, batch_first=True
            ),
            num_layers=self.num_layers
        ).double()
        self.fc = nn.Linear(self.dim_feedforward, self.vocab_size).double()

    def forward(self, src, tgt):
        encoded_src = self.encode(src)
        decoded_out = self.decode(tgt, encoded_src)
        return decoded_out

    def encode(self, src):
        # src = self.embedding(src) # TODO: normalize by d_model?
        # TODO: positional encoding
        src = self.encoder(src)
        return src

    def decode(self, tgt, encoded_src):
        tgt = self.embedding(tgt).double() # TODO: normalize by d_model?
        # TODO: positional encoding, mask
        out = self.decoder(tgt, encoded_src)
        out = self.fc(out)
        return out

    def predict(self, src, use_beam_search=True):
        '''
        Note: transformer can only currenty predict one phrase at a time; possible to
        modify to do batched prediction
        '''
        if use_beam_search:
            # https://towardsdatascience.com/temperature-scaling-and-beam-search-text-generation-in-llms-for-the-ml-adjacent-21212cc5dddb#45c4
            encoded_src = self.encode(src)
            beam_width = 3

            # The initial candidate sequence is simply the start token ID with
            # a sequence score of 0
            # Sequences should be tuples of (sequence tensor, sequence score)
            candidate_sequences = [
                (torch.ones((1,1)).fill_(self.tokenizer.cls_token_id).to(torch.int64), -np.infty)
            ]

            for i in tqdm.tqdm(range(self.max_output_length)):
                # Temporary list to store candidates for the next generation step
                next_step_candidates = []

                # Iterate through all candidate sequences; for each, generate the next
                # most likely tokens and add them to the next-step sequnce of candidates
                for candidate, candidate_score in candidate_sequences:

                    # skip candidate sequences which have included the end-of-sequence token
                    if candidate[-1,-1] != self.tokenizer.sep_token_id:

                        # Predict next token
                        output = self.decode(tgt=candidate, encoded_src=encoded_src)

                        # # Extract logits from output
                        # logits = output.logits[:, -1, :]

                        # # Scale logits using temperature value
                        # scaled_logits = logits / temperature

                        # Construct probability distribution against scaled
                        # logits through softmax activation function
                        probs = torch.softmax(output[:,-1], dim=1).squeeze()

                        # Select top k (beam_width) probabilities and IDs from the distribution
                        top_probs, top_ids = probs.topk(beam_width)

                        # For each of the top-k generated tokens, append to this
                        # candidate sequence, update its score, and append to the list of next
                        # step candidates
                        for i in range(beam_width):
                            # the new token ID
                            next_token_id = top_ids[i].item()

                            # log-prob of the above token
                            next_score = torch.log(top_probs[i]).item()

                            next_seq = deepcopy(candidate)

                            # Adds the new token to the end of this sequence, and updates its
                            # raw and normalized scores. Scores are normalized by sequence token
                            # length, to avoid penalizing longer sequences
                            next_seq = torch.concat((next_seq, torch.tensor([[next_token_id]])), dim=-1)

                            # Append the updated sequence to the next candidate sequence set
                            next_step_candidates.append((next_seq, next_score))
                    else:
                        # Append the canddiate sequence as-is to the next-step candidates
                        # if it already contains an end-of-sequence token
                        next_step_candidates.append((candidate, candidate_score))

                # Sort the next-step candidates by their score, select the top-k
                # (beam_width) scoring sequences and make them the new
                # candidate_sequences list
                next_step_candidates.sort(key=lambda tup: tup[1], reverse=True) # sort by score
                candidate_sequences = next_step_candidates[:beam_width]

                # Break if all sequences in the heap end with the eos_token_id
                if all(cand[-1,-1] == self.tokenizer.sep_token_id for cand, _ in candidate_sequences):
                    break

            return '\n\n'.join(' '.join(self.tokenizer.batch_decode(cand)) for cand, _ in candidate_sequences)

        else:
            encoded_src = self.encode(src)
            output_ids = torch.ones((1,1)).fill_(self.tokenizer.cls_token_id).to(torch.int64)

            # output_ids = torch.cat((output_ids, torch.tensor([self.tokenizer.encode('imagine', add_special_tokens=False)])), dim=1)
            for i in range(self.max_output_length - 1):
                out = self.decode(output_ids, encoded_src)
                _, output_id = torch.max(out[:, -1], dim=1)
                output_id = output_id.item()
                output_ids = torch.cat((output_ids, torch.ones((1,1)).fill_(output_id)), dim=1).to(torch.int64)
                if output_id == self.tokenizer.sep_token_id:
                    break

            return ' '.join(self.tokenizer.batch_decode(output_ids))

class ConvTransformer(nn.Module):
    def __init__(self, d_model, tokenizer, max_output_length=64):
        super().__init__()
        self.d_model = d_model
        self.tokenizer = tokenizer
        self.transformer = Transformer(d_model, tokenizer, max_output_length)
        self.max_output_length = max_output_length

    def convolution(self, conv_input):
        # conv_input shape: (batch_size, num_features, seq_len)
        self.conv_config = {
            'in_channels': conv_input.shape[1],
            'out_channels': self.d_model,
            'kernel_size': 3,
            'stride': 1,
            'padding': 1,
            'dilation': 1,
            'bias': False
        }
        conv = nn.Sequential(
            nn.Conv1d(**self.conv_config),
            nn.ReLU(),
            # nn.BatchNorm1d(64)
        ).double()
        conv_out = conv(conv_input)
        return conv_out


    def forward(self, feature_tensor, target_text):
        # feature_tensor shape: (batch_size, seq_len, num_features)
        conv_input = feature_tensor.permute(0, 2, 1) # conv requires shape (batch_size, num_features, seq_len)
        conv_output = self.convolution(conv_input)
        transformer_input = conv_output.permute(0, 2, 1) # return shape to (batch_size, seq_len, num_features)
        return self.transformer(transformer_input, target_text)

    def predict(self, feature_tensor, use_beam_search=True):
        # start with transformer implementation, then improve
        conv_input = feature_tensor.permute(0, 2, 1)
        conv_output = self.convolution(conv_input)
        transformer_input = conv_output.permute(0, 2, 1)

        return self.transformer.predict(transformer_input, use_beam_search=use_beam_search)



def train(model, device, train_loader, val_loader=None, num_epochs=20, loss_fn=nn.CrossEntropyLoss, optimizer=torch.optim.Adam,
          lr=5e-4, weight_decay=1e-2):

        loss_fn = loss_fn(
            ignore_index=model.tokenizer.pad_token_id, # ignore padding token for loss
        )

        optimizer = optimizer(model.parameters(), lr, weight_decay=weight_decay)

        train_losses = []
        val_losses = []

        model.to(device)
        loss_fn.to(device)

        # Train the model
        for epoch in range(num_epochs):
            model.train()
            train_step_loss = []
            for i, (features, target) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
                features, target = features.to(device), target.to(device)
                out = model(features, target[:, :-1]) # autoregressive target
                loss = loss_fn(
                    out.contiguous().view(-1, len(model.tokenizer)),
                    target[:, 1:].contiguous().view(-1), # flatten for loss, target shifted right
                )

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                train_step_loss.append(loss.item())

            train_losses.append(np.mean(train_step_loss))

            # Evaluate loss on validation set
            model.eval()
            with torch.no_grad():
                val_step_loss = []
                for i, (features, target) in tqdm.tqdm(enumerate(val_loader), total=len(val_loader)):
                    features, target = features.to(device), target.to(device)
                    out = model(features, target[:, :-1])
                    loss = loss_fn(
                        out.contiguous().view(-1, len(model.tokenizer)),
                        target[:, 1:].contiguous().view(-1),
                    )

                    val_step_loss.append(loss.item())
            val_losses.append(np.mean(val_step_loss))

            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {np.mean(train_step_loss)} / {np.mean(val_step_loss)}')

        return train_losses, val_losses

def eval(model, dataset):
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
        # rescale_with_baseline=True
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



def run_offline(audio_data, pose_data, force_data, tokenizer, from_audio_json=False):
    print('start run offline...')
    aligned_dataset = AlignedDataset(audio_data, pose_data, force_data, tokenizer,
                                     from_audio_json=from_audio_json)
    print(f'{aligned_dataset[0][0].shape=}')
    train_set, val_set = random_split(aligned_dataset, [0.8, 0.2])
    train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=4, shuffle=True)


    # TODO: train/test split
    print('data preprocessed...')

    ### initialize new model ###
    conv_transformer = ConvTransformer(D_MODEL, tokenizer)

    # ### training ###
    # print('start transformer training...')
    # if torch.cuda.is_available():
    #     device = torch.device('cuda')
    # else:
    #     device = torch.device('cpu')
    # train_losses, val_losses = train(conv_transformer, device, train_loader, val_loader, num_epochs=30) # train model

    # ### plot losses ###
    # plt.plot(np.arange(len(train_losses)), train_losses, label='training loss')
    # plt.plot(np.arange(len(val_losses)), val_losses, label='validation loss')
    # plt.legend()

    ### load saved model ###
    print('load transformer...')
    model_name = f'{MODELS_FP}/model_phrase_v1.3'
    conv_transformer.load_state_dict(torch.load(model_name)) # load model weights, testing

    ### inference ###
    print('start inference...')
    for _ in range(10):
        # start = time.time()
        ix = random.randint(0, len(aligned_dataset)-1)
        print(f'\nix: {ix}')
        src, tgt = aligned_dataset[ix]
        decoded_output = conv_transformer.predict(src.unsqueeze(0), use_beam_search=False)
        print(f'Expected: {tokenizer.decode(tgt[:32])}')
        print(f'Predicted: {decoded_output}')
        # print(f'Decoding time elapsed: {time.time() - start}')

    # ### evaluate ###
    # eval(conv_transformer, aligned_dataset)

    # ### save trained model ###
    # model_name = f'{MODELS_FP}/model_phrase_v1.2'
    # torch.save(conv_transformer.state_dict(), model_name)
    # plt.savefig(model_name + '.png')

    # plt.show()

# def run_online(transformer_model, tokenizer, delta_t=0.2):
#     print('Starting online...')
#     collect_interval = 1 # seconds

#     q = mp.Queue()
#     proc_collect = mp.Process(target=collect_poses, args=(q,))
#     proc_collect.start()

#     t_interval_start = None
#     interval_records = {
#         'timestamp': [],
#         'skeletons': []
#     }
#     print('Enter collection loop...')
#     while True:
#         timestamp, pose_data = q.get()
#         # t_interval_start = first timestamp of the data collection interval
#         t_interval_start = timestamp // delta_t * delta_t if t_interval_start is None else t_interval_start

#         # if timestamp within the interval, append it for future processing
#         if timestamp < t_interval_start + collect_interval:
#             # print(f'{len(interval_records['timestamp'])=}')
#             interval_records['timestamp'].append(timestamp)
#             interval_records['skeletons'].append(pose_data)
#         # else, process the interval's collected data for transformer decoding
#         else:
#             print('Processing...')
#             # print(f'{interval_records["skeletons"][-1]=}')
#             # pose_df = pd.DataFrame.from_records(interval_records)
#             # HOTFIX
#             audio_fp = 'vincent-dev\data\lightbuzz_table_1\cut_audio.wav'
#             force_fp = 'vincent-dev\data\lightbuzz_table_1\cut_data.csv'
#             feature_tensor, text_tensor = preprocess_pipeline(audio_fp, interval_records, force_fp, tokenizer, from_records=True)
#             conv_input = feature_tensor.permute(1,0) # conv requires shape (C_in, L_in)
#             print(f'{conv_input.shape=}')
#             conv_output = convolution(conv_input, D_MODEL)

#             transformer_input = conv_output.transpose(1,0)
#             # transformer_target = text_tensor.squeeze()
#             print(f'{transformer_input.shape=}')

#             preds = transformer_model.predict(transformer_input)
#             print(preds)


#             # reset interval records
#             interval_records = {
#                 'timestamp': [],
#                 'skeletons': []
#             }
#             t_interval_start = None
#     proc_collect.terminate()

# def test_online():
#     q = mp.Queue()
#     proc_collect = mp.Process(target=collect_poses, args=(q,))
#     proc_collect.start()

#     records = {
#         'timestamp': [],
#         'skeletons': []
#     }

#     while True:
#         timestamp, pose_data = q.get()
#         records['timestamp'].append(timestamp)
#         records['skeletons'].append(pose_data)
#         if len(records['timestamp']) == 100:
#             break

#     pose_df = pd.DataFrame.from_records(records)
#     # pose_df.to_csv('server_output.csv')
#     print(pose_df.dtypes)
#     # print(set(pose_df['skeletons'][0][0].keys()))

#     # for record in records:
#     #     with open('vincent-dev\server_output.jsonl', 'a') as f:
#     #         f.write(json.dumps(record))


#     proc_collect.terminate()


if __name__ == '__main__':

    file_range = range(1,7)
    audio_fp_list, phrases_fp_list, pose_fp_list, force_fp_list = [], [] ,[], []
    for i in file_range:
        audio_fp_list.append(f'data/lightbuzz_table_{i}/cut_audio.wav')
        phrases_fp_list.append(f'data/lightbuzz_table_{i}/phrases.json')
        pose_fp_list.append(f'data/lightbuzz_table_{i}/cut_poses.jsonl')
        force_fp_list.append(f'data/lightbuzz_table_{i}/cut_data.csv')

    tokenizer = transformers.BertTokenizerFast.from_pretrained('bert-base-uncased')

    # run_offline(audio_json_list, pose_fp_list, force_fp_list, tokenizer)
    run_offline(phrases_fp_list, pose_fp_list, force_fp_list, tokenizer)


    # model = Transformer(D_MODEL, tokenizer)
    # model.load_state_dict(torch.load('vincent-dev\models\model_v0_1'))
    # run_online(model, tokenizer)


    # test_online()


    # aligned_dataset = AlignedDataset(audio_json_list, pose_fp_list, force_fp_list, tokenizer, from_audio_json=True)
    # aligned_dataloader = DataLoader(aligned_dataset, batch_size=16, shuffle=False)
    # model = ConvTransformer(D_MODEL, tokenizer)
    # for f, t in aligned_dataloader:
    #     print(model(f, t))
    #     break