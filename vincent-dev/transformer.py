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
from lightbuzz_poses import collect_poses


### global constants ###
DATA_FP = 'data'
D_MODEL = 64
MODELS_FP = 'models'

class CombinedDataset(Dataset):
    def __init__(self, input_tensor, target_tensor):
        self.input = input_tensor.detach().clone()
        self.target = target_tensor.detach().clone()

        assert self.input.shape[0] == self.target.shape[0] # tensors are of the same length

    def __len__(self):
        return self.input.shape[0]

    def __getitem__(self, ix):
        return self.input[ix], self.target[ix]

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




def train(model, device, train_loader, val_loader=None, num_epochs=20, loss_fn=nn.CrossEntropyLoss, optimizer=torch.optim.Adam,
          lr=0.001, weight_decay=0):

        loss_fn = loss_fn()
        optimizer = optimizer(model.parameters(), lr, weight_decay=weight_decay)

        lossi = []
        val_lossi = []

        model.to(device)
        loss_fn.to(device)

        # Train the model
        for epoch in range(num_epochs):
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

def run_offline():
    ### read and preprocess data ###
    audio_fp_list = [
        f'{DATA_FP}/lightbuzz_table_{i}/cut_audio.wav'
        for i in range(1,7)
    ]
    pose_fp_list = [
        f'{DATA_FP}/lightbuzz_table_{i}/cut_poses.jsonl'
        for i in range(1,7)
    ]
    force_fp_list = [
        f'{DATA_FP}/lightbuzz_table_{i}/cut_data.csv'
        for i in range(1,7)
    ]
    tokenizer = transformers.BertTokenizerFast.from_pretrained('bert-base-uncased')

    conv_input, text_tensor = preprocess_pipeline(audio_fp_list, pose_fp_list, force_fp_list, tokenizer)
    print(f'{conv_input.shape=}')

    ### convolution ###
    conv_output = convolution(conv_input, D_MODEL)

    ### transformer ###
    # transformer_config = {
    #     'd_model': D_MODEL,
    #     'nhead': 8,
    #     'num_encoder_layers': 6,
    #     'num_decoder_layers': 6,
    #     'batch_first': True,
    # }

    transformer_model = Transformer(D_MODEL, tokenizer)
    transformer_target = text_tensor.squeeze()
    transformer_input = conv_output.transpose(1,0)





    ### dataset ###
    dataset = CombinedDataset(transformer_input, transformer_target)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=False)

    ### training ###
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # train(transformer_model, device, train_loader) # train model
    transformer_model.load_state_dict(torch.load(f'{MODELS_FP}/model_v0')) # load model weights, testing

    start = time.time()
    decoded_output = transformer_model.predict(transformer_input)
    print(decoded_output)
    print(f'Decoding time elapsed: {time.time() - start}')

    # torch.save(transformer_model.state_dict(), f'{MODELS_FP}/model_v0_1')

def run_online(transformer_model, delta_t=0.2):
    collect_interval = 1 # seconds

    q = mp.Queue()
    proc_collect = mp.Process(target=collect_poses, args=(q,))
    proc_collect.start()

    t_interval_start = None
    interval_records = {
        'timestamp': [],
        'skeletons': []
    }
    while True:
        timestamp, pose_data = q.get()
        # t_interval_start = first timestamp of the data collection interval
        t_interval_start = timestamp if t_interval_start is None else t_interval_start

        # if timestamp within the interval, append it for future processing
        if timestamp < t_interval_start + collect_interval:
            interval_records['timestamp'].append(timestamp)
            interval_records['skeletons'].append(pose_data)
        # else, process the interval's collected data for transformer decoding
        else:
            pose_df = pd.DataFrame.from_records(interval_records)
            conv_input, text_tensor = preprocess_pipeline(pose_df)
            conv_output = convolution(conv_input, D_MODEL)

            transformer_input = conv_output.transpose(1,0)
            # transformer_target = text_tensor.squeeze()

            preds = transformer_model.predict(transformer_input)
            print(preds)


            # reset interval records
            interval_records = {
                'timestamp': [],
                'skeletons': []
            }



if __name__ == '__main__':
    run_offline()

