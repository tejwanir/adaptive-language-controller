import json
import multiprocessing as mp
import numpy as np
import pandas as pd
import time
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import tqdm
import transformers

from data_processing import preprocess_pipeline
from lightbuzz_poses_transformer import collect_poses


### global constants ###
DATA_FP = 'data'
D_MODEL = 64
MODELS_FP = 'models'


class AlignedDataset(Dataset):
    def __init__(self, audio_data, pose_data, force_data, tokenizer,
                 from_records=False, from_audio_json=False):
        feature_tensor, text_tensor = preprocess_pipeline(audio_data, pose_data, force_data, tokenizer,
                                                          from_records=from_records, from_audio_json=from_audio_json)
        self.feature_tensor = feature_tensor
        self.text_tensor = text_tensor.squeeze()

    def __len__(self):
        return self.feature_tensor.shape[0]

    def __getitem__(self, ix):
        return self.feature_tensor[ix], self.text_tensor[ix]

class Transformer(nn.Module):
    def __init__(self, d_model, tokenizer, max_output_length=1):
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
                self.d_model, self.nhead, self.dim_feedforward
            ),
            num_layers=self.num_layers
        ).double()
        self.fc = nn.Linear(self.dim_feedforward, self.vocab_size).double()

    def forward(self, src, tgt):
        encoded_src = self.encode(src)
        out = self.decode(tgt, encoded_src)
        return out

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

    def predict(self, src):
        encoded_src = self.encode(src)

        tgt = torch.ones((src.shape[0])).int()
        tgt = tgt * 101 # TODO: hardcoded start token

        # tgt = torch.randint(2000, 3000, (src.shape[0],)).int()
        out = self.decode(tgt, encoded_src)
        out = torch.softmax(out, dim=1)
        out = torch.argmax(out, dim=1)
        return self.tokenizer.batch_decode(out)

class ConvTransformer(nn.Module):
    def __init__(self, d_model, tokenizer, max_output_length=1):
        super().__init__()
        self.d_model = d_model
        self.tokenizer = tokenizer
        self.transformer = Transformer(d_model, tokenizer, max_output_length)

    def convolution(self, conv_input):
        self.conv_config = {
            'in_channels': conv_input.shape[0],
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
        return conv(conv_input)

    def forward(self, feature_tensor, target_text):
        conv_input = feature_tensor.permute(1,0) # conv requires shape (C_in, L_in)
        conv_output = self.convolution(conv_input)
        transformer_input = conv_output.transpose(1,0)
        return self.transformer(transformer_input, target_text)

    def predict(self, feature_tensor):
        # start with transformer implementation, then improve
        conv_input = feature_tensor.permute(1,0) # conv requires shape (C_in, L_in)
        conv_output = self.convolution(conv_input)
        transformer_input = conv_output.transpose(1,0)

        encoded_src = self.transformer.encode(transformer_input)

        tgt = torch.ones((transformer_input.shape[0])).int()
        tgt = tgt * 101 # TODO: hardcoded start token

        # tgt = torch.randint(2000, 3000, (transformer_input.shape[0],)).int()
        out = self.transformer.decode(tgt, encoded_src)
        out = torch.softmax(out, dim=1)
        out = torch.argmax(out, dim=1)
        return self.tokenizer.batch_decode(out)



def train(model, device, train_loader, val_loader=None, num_epochs=20, loss_fn=nn.CrossEntropyLoss, optimizer=torch.optim.Adam,
          lr=0.001, weight_decay=0):

        loss_fn = loss_fn()
        optimizer = optimizer(model.parameters(), lr, weight_decay=weight_decay)

        lossi = []
        val_lossi = []

        model.to(device)
        loss_fn.to(device)

        # Train the model
        for epoch in tqdm.tqdm(range(num_epochs)):
            model.train()
            epoch_train_loss = 0
            step_loss = []
            for i, (features, target) in tqdm.tqdm(enumerate(train_loader)):
                features, target = features.to(device), target.to(device)
                out = model(features, target)
                loss = loss_fn(out, target)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                step_loss.append(loss.item())
                epoch_train_loss += loss.item()

            lossi.append(np.mean(step_loss))

            # Evaluate loss on validation set
            model.eval()
            epoch_val_loss = 0
            # with torch.no_grad():
            #     step_loss = []
            #     for i, (data, target) in enumerate(val_loader):
            #         data, target = data.to(device), target.to(device)
            #         out = model(data)
            #         loss = loss_fn(out, target)

            #         step_loss.append(loss.item())
            #         epoch_val_loss += loss.item()
            # val_lossi.append(np.mean(step_loss))

            # print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_train_loss / len(train_loader)} / {epoch_val_loss / len(val_loader)}')
            # print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_train_loss / len(train_loader)}')

        return lossi, val_lossi

def convolution(conv_input, d_model):
    conv_config = {
        'in_channels': conv_input.shape[0],
        'out_channels': d_model,
        'kernel_size': 3,
        'stride': 1,
        'padding': 1,
        'dilation': 1,
        'bias': False
    }

    conv_model = nn.Sequential(
        nn.Conv1d(**conv_config),
        nn.ReLU(),
        # nn.BatchNorm1d(64), # debug
        # nn.Flatten()
    )
    conv_model.double()
    conv_output = conv_model(conv_input)
    return conv_output


def run_offline(audio_data, pose_data, force_data, tokenizer, from_audio_json=False):
    print('start run offline...')
    aligned_dataset = AlignedDataset(audio_data, pose_data, force_data, tokenizer,
                                     from_audio_json=from_audio_json)
    aligned_dataloader = DataLoader(aligned_dataset, batch_size=64, shuffle=False)
    # TODO: train/test split
    print('data preprocessed...')

    ### model ###
    conv_transformer = ConvTransformer(D_MODEL, tokenizer)

    ### training ###
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print('start transformer training...')
    train(conv_transformer, device, aligned_dataloader) # train model
    # transformer_model.load_state_dict(torch.load(f'{MODELS_FP}/model_v0')) # load model weights, testing

    ### inference ###
    start = time.time()
    decoded_output = conv_transformer.predict(aligned_dataset.feature_tensor)
    print(decoded_output)
    print(f'Decoding time elapsed: {time.time() - start}')

    # torch.save(transformer_model.state_dict(), f'{MODELS_FP}/model_v0_1')

def run_online(transformer_model, tokenizer, delta_t=0.2):
    print('Starting online...')
    collect_interval = 1 # seconds

    q = mp.Queue()
    proc_collect = mp.Process(target=collect_poses, args=(q,))
    proc_collect.start()

    t_interval_start = None
    interval_records = {
        'timestamp': [],
        'skeletons': []
    }
    print('Enter collection loop...')
    while True:
        timestamp, pose_data = q.get()
        # t_interval_start = first timestamp of the data collection interval
        t_interval_start = timestamp // delta_t * delta_t if t_interval_start is None else t_interval_start

        # if timestamp within the interval, append it for future processing
        if timestamp < t_interval_start + collect_interval:
            # print(f'{len(interval_records['timestamp'])=}')
            interval_records['timestamp'].append(timestamp)
            interval_records['skeletons'].append(pose_data)
        # else, process the interval's collected data for transformer decoding
        else:
            print('Processing...')
            # print(f'{interval_records["skeletons"][-1]=}')
            # pose_df = pd.DataFrame.from_records(interval_records)
            # HOTFIX
            audio_fp = 'vincent-dev\data\lightbuzz_table_1\cut_audio.wav'
            force_fp = 'vincent-dev\data\lightbuzz_table_1\cut_data.csv'
            feature_tensor, text_tensor = preprocess_pipeline(audio_fp, interval_records, force_fp, tokenizer, from_records=True)
            conv_input = feature_tensor.permute(1,0) # conv requires shape (C_in, L_in)
            print(f'{conv_input.shape=}')
            conv_output = convolution(conv_input, D_MODEL)

            transformer_input = conv_output.transpose(1,0)
            # transformer_target = text_tensor.squeeze()
            print(f'{transformer_input.shape=}')

            preds = transformer_model.predict(transformer_input)
            print(preds)


            # reset interval records
            interval_records = {
                'timestamp': [],
                'skeletons': []
            }
            t_interval_start = None
    proc_collect.terminate()

def test_online():
    q = mp.Queue()
    proc_collect = mp.Process(target=collect_poses, args=(q,))
    proc_collect.start()

    records = {
        'timestamp': [],
        'skeletons': []
    }

    while True:
        timestamp, pose_data = q.get()
        records['timestamp'].append(timestamp)
        records['skeletons'].append(pose_data)
        if len(records['timestamp']) == 100:
            break

    pose_df = pd.DataFrame.from_records(records)
    # pose_df.to_csv('server_output.csv')
    print(pose_df.dtypes)
    # print(set(pose_df['skeletons'][0][0].keys()))

    # for record in records:
    #     with open('vincent-dev\server_output.jsonl', 'a') as f:
    #         f.write(json.dumps(record))


    proc_collect.terminate()


if __name__ == '__main__':
    file_range = range(1,3)
    audio_fp_list = [
        f'{DATA_FP}/lightbuzz_table_{i}/cut_audio.wav'
        for i in file_range
    ]
    audio_json_list = [
        f'{DATA_FP}/lightbuzz_table_{i}/transcription_base.json'
        for i in file_range
    ]
    pose_fp_list = [
        f'{DATA_FP}/lightbuzz_table_{i}/cut_poses.jsonl'
        for i in file_range
    ]
    force_fp_list = [
        f'{DATA_FP}/lightbuzz_table_{i}/cut_data.csv'
        for i in file_range
    ]
    tokenizer = transformers.BertTokenizerFast.from_pretrained('bert-base-uncased')


    run_offline(audio_json_list, pose_fp_list, force_fp_list, tokenizer, from_audio_json=True)


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