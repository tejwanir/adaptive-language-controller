import tensorflow as tf
import typing
from pathlib import Path

import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from filter_poses import FrameCorrector, PoseFilter, UserPoses, preprocess_poses
from lightbuzz_poses import collect_poses


from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Embedding, LSTM, Dense 
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences 
import regex as re 

def run_admittance_trajectory():
    from admittance import admittance_loop
    from global_constants import slow_turn, slow_walk
    from paths import Movement
    from ur5 import gripper, rtde_c

    movement = Movement([slow_turn, slow_walk, slow_turn, slow_walk, slow_turn])
    gripper.move_and_wait_for_pos(105, 255, 1)
    rtde_c.zeroFtSensor()
    init_pose = movement.start_pose()
    init_info = init_pose.tolist() + [0] * 6
    robot_info = mp.Array("d", init_info)
    admittance_loop([movement], robot_info)

def file_to_sentence_list(file_path): 
    with open(file_path, 'r') as file: 
        text = file.read() 
  
    # Splitting the text into sentences using 
    # delimiters like '.', '?', and '!' 
    sentences = [sentence.strip() for sentence in re.split( 
        r'(?<=[.!?])\s+', text) if sentence.strip()] 
  
    return sentences 

file_path = 'data.txt'
text_data = file_to_sentence_list(file_path) 
  
# Tokenize the text data 
tokenizer = Tokenizer() 
tokenizer.fit_on_texts(text_data) 
total_words = len(tokenizer.word_index) + 1
  
# Create input sequences 
input_sequences = [] 
for line in text_data: 
    token_list = tokenizer.texts_to_sequences([line])[0] 
    for i in range(1, len(token_list)): 
        n_gram_sequence = token_list[:i+1] 
        input_sequences.append(n_gram_sequence) 
  
# Pad sequences and split into predictors and label 
max_sequence_len = max([len(seq) for seq in input_sequences]) 
input_sequences = np.array(pad_sequences( 
    input_sequences, maxlen=max_sequence_len, padding='pre')) 
X, y = input_sequences[:, :-1], input_sequences[:, -1] 
  
# Convert target data to one-hot encoding 
y = tf.keras.utils.to_categorical(y, num_classes=total_words) 

KNN_INTERVAL=1

def run_admittance_trajectory():
    from admittance import admittance_loop
    from global_constants import slow_turn, slow_walk
    from paths import Movement
    from ur5 import gripper, rtde_c

    movement = Movement([slow_turn, slow_walk, slow_turn, slow_walk, slow_turn])
    gripper.move_and_wait_for_pos(105, 255, 1)
    rtde_c.zeroFtSensor()
    init_pose = movement.start_pose()
    init_info = init_pose.tolist() + [0] * 6
    robot_info = mp.Array("d", init_info)
    admittance_loop([movement], robot_info)


def run_admittance_trajectory():
    from admittance import admittance_loop
    from global_constants import slow_turn, slow_walk
    from paths import Movement
    from ur5 import gripper, rtde_c

    movement = Movement([slow_turn, slow_walk, slow_turn, slow_walk, slow_turn])
    gripper.move_and_wait_for_pos(105, 255, 1)
    rtde_c.zeroFtSensor()
    init_pose = movement.start_pose()
    init_info = init_pose.tolist() + [0] * 6
    robot_info = mp.Array("d", init_info)
    admittance_loop([movement], robot_info)


def solve():
    nn_model = tf.keras.models.load_model('nnmodel.keras')
    rnn_model = tf.keras.models.load_model('model.keras')
    import torch
    from scipy.spatial import KDTree
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from sound import AsyncTTSPlayer

    device = "cuda"
    model_id = "openai-community/gpt2"
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)


    q: "mp.Queue[tuple[float, UserPoses]]" = mp.Queue()
    proc_collect = mp.Process(target=collect_poses, args=(q,))
    proc_collect.start()
    proc_move = mp.Process(target=run_admittance_trajectory, args=())
    proc_move.start()

    all_timestamp_poses = []
    idx_prev = 0
    tts_player = AsyncTTSPlayer()
    last_timestamp = 0
    knn_check_interval = 1 / 10


    past_pos = [0,0,0]
    context = "1"
    speech = ""
    while True: 
        token_list = tokenizer.texts_to_sequences([context])[0] 
        token_list = pad_sequences( 
            [token_list], maxlen=max_sequence_len-1, padding='pre') 
        predicted_probs = rnn_model.predict(token_list) 
        predicted_word = tokenizer.index_word[np.argmax(predicted_probs)] 
        context += " " + predicted_word 
        if predicted_word == "stop":
            break
        speech += predicted_word

    if tts_player.ready.is_set():
        tts_player.put_text(speech)
        tts_player.ready.clear()

    while True:
        timestamp, poses = q.get()
        feature = lambda p: np.array(p[0]["WristLeft"])

        try:
            pos_now = feature(poses)
        except KeyError:
            continue

        while (
            idx_prev < len(all_timestamp_poses)
            and all_timestamp_poses[idx_prev][0] < timestamp - KNN_INTERVAL
        ):
            idx_prev += 1
        if (
            idx_prev < len(all_timestamp_poses)
            and timestamp - last_timestamp >= knn_check_interval
        ):
            predicted_word += nn_model.predict(np.array([past_pos[0],past_pos[1],past_pos[2],pos_now[0],pos_now[1],pos_now[2]]))
            speech = ""
            while True: 
                token_list = tokenizer.texts_to_sequences([context])[0] 
                token_list = pad_sequences( 
                    [token_list], maxlen=max_sequence_len-1, padding='pre') 
                predicted_probs = rnn_model.predict(token_list) 
                predicted_word = tokenizer.index_word[np.argmax(predicted_probs)] 
                context += " " + predicted_word 
                if predicted_word == "stop":
                    break
                speech += predicted_word
            if tts_player.ready.is_set():
                tts_player.put_text(speech)
                tts_player.ready.clear()
            

            
            

if __name__ == "__main__":
    solve()
    pass