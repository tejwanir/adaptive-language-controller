import json
import numpy as np
import pandas as pd
import re
import torch
import transformers
import whisper

from util import NumpyEncoder



### global constants ###

DELTA_T = 0.2
DATA_FP = 'data'
KEEP_JOINTS = {
    'ShoulderRight',
    'ElbowRight',
    'WristRight',
    'ShoulderLeft',
    'ElbowLeft',
    'WristLeft',
}
KEEP_MEASUREMENTS_DICT = {
    'pos2D': 2,
    'pos3D': 3,
    'orientation': 4
}


### audio/text processing ###

def get_transcription(audio_fp, whisper_model_name='base'):
    model = whisper.load_model(whisper_model_name)
    transcription = model.transcribe(audio_fp, word_timestamps=True, fp16=False)
    return transcription

def bucketize_text_data(transcription, delta_t=DELTA_T):
    d = {
        'timestamps': [],
        'text': [],
    }

    # TODO: add silence token, account for silence mid-speech; make sure to adjust embedding size
    # https://huggingface.co/docs/transformers/v4.39.3/en/main_classes/tokenizer#transformers.PreTrainedTokenizer.add_special_tokens

    all_word_data = [word_data for segment in transcription['segments'] for word_data in segment['words']]
    word_ix = 0
    curr_time = 0
    while word_ix < len(all_word_data):
        word_data = all_word_data[word_ix]
        t_start, t_end = word_data['start'], word_data['end']
        # if word occurs at curr_time, add word to text (i.e., assign word to curr_time bucket)
        if t_start <= curr_time and curr_time < t_end:
            d['text'].append(word_data['word'])
            curr_time += delta_t
        # else, test if silence occurs during curr_time, then proceed to next word
        else:
            # if next bucket occurs before the next word, then next bucket should be assigned silence
            while word_ix+1 < len(all_word_data) and curr_time < all_word_data[word_ix+1]['start']:
                d['text'].append('[PAD]') # TODO: replace with actual silence token instead of [PAD]
                curr_time += delta_t
            word_ix += 1
    d['timestamps'] = np.arange(0, round(curr_time, 1), delta_t) # TODO: make a more robust fix for float precision

    assert len(d['timestamps']) == len(d['text'])
    return pd.DataFrame.from_dict(d)

def tokenize_text_data(text_df, tokenizer):
    text_df['text'] = (text_df['text'].apply(lambda text: tokenizer.encode(text, add_special_tokens=False)[0])
                                      .astype('int64'))
    return text_df

def preprocess_text_data(audio_fp, tokenizer):
    transcription = get_transcription(audio_fp, whisper_model_name='base')
    text_df = bucketize_text_data(transcription)
    text_df = tokenize_text_data(text_df, tokenizer)
    return text_df


### pose data processing ###

def filter_skeleton(skeletons_list, user_id, keep_joints=KEEP_JOINTS):
    '''
    Filter the pose data to contain only the user specified by `user_id`.
    Filter `user_id` skeleton to contain only desired joints (defined by keep_joints).
    Return dicitonary object of remaining pose data.

    pose_df['skeletons'] is a list of json objects, with each element
    corresponding to a different "user" from the original video
    '''
    # HOTFIX
    if isinstance(skeletons_list, list):
        for skeleton in skeletons_list:
            # filter by user_id
            if 'user_id' in skeleton:
                if skeleton['user_id'] == user_id:
                    # filter by desired joints
                    return {joint: skeleton[joint] for joint in skeleton if joint in keep_joints}
    else:
        skeleton = skeletons_list[0]
        print(f'{keep_joints=}')
        ret = {joint: skeleton[joint] for joint in skeleton if joint in keep_joints}
        return ret

def bucketize_pose_data(pose_df, delta_t=DELTA_T, keep_joints=KEEP_JOINTS,
                        keep_measurements_dict=KEEP_MEASUREMENTS_DICT):
    '''
    Re-discretize the pose data into more regular time buckets for the purpose
    of easier temporal alignment with force data, text data, etc.

    Current strategy: use a fixed time interval between time buckets, with each
    bucket containing the averaged data of each timestamp in the range
    [bucket_t, bucket_t + delta_t) for bucket start time bucket_t. Averaged
    measurements are defined by keep_measurements
    '''
    # start_t = 1707358708.6 # HARDCODED
    start_t = pose_df['timestamp'][0] // delta_t * delta_t
    curr_bucket = start_t

    records = []

    curr_record = {
        'timestamp': curr_bucket,
        'skeletons': {
            joint: {
                measurement: np.zeros(keep_measurements_dict[measurement])
                for measurement in keep_measurements_dict
            } for joint in keep_joints
        }
    }

    bucket_count = 0
    for ix, row in pose_df.iterrows():
        if row['timestamp'] < curr_bucket + delta_t:
            # sum up data for each measurement, for each joint
            for joint in keep_joints:
                for measurement in keep_measurements_dict:
                    curr_record['skeletons'][joint][measurement] += np.array(row['skeletons'][joint][measurement])
            bucket_count += 1

        else:
            # divide accumulated coordinates by bucket_count to get average of data
            for joint in keep_joints:
                for measurement in keep_measurements_dict:
                    assert bucket_count > 0
                    curr_record['skeletons'][joint][measurement] /= bucket_count

            # append curr_record to records
            records.append(curr_record)

            # update curr_bucket, create new curr_record, and reset bucket_count
            curr_bucket += delta_t
            curr_record = {
                'timestamp': curr_bucket,
                'skeletons': {
                    joint: {
                        measurement: np.zeros(keep_measurements_dict[measurement])
                        for measurement in keep_measurements_dict
                    } for joint in keep_joints
                }
            }
            bucket_count = 0

    return pd.DataFrame.from_records(records)

def bucketize_pose_data_measurementless(pose_df, delta_t=DELTA_T, keep_joints=KEEP_JOINTS,
                        keep_measurements_dict=KEEP_MEASUREMENTS_DICT):
    '''
    bucketizee measurementless
    '''
    # start_t = 1707358708.6 # HARDCODED
    start_t = pose_df['timestamp'][0] // delta_t * delta_t
    curr_bucket = start_t

    records = []

    curr_record = {
        'timestamp': curr_bucket,
        'skeletons': {
            joint: np.zeros(3) for joint in keep_joints
        }
    }

    bucket_count = 0
    for ix, row in pose_df.iterrows():
        if row['timestamp'] < curr_bucket + delta_t:
            # sum up data for each measurement, for each joint
            for joint in keep_joints:
                curr_record['skeletons'][joint] += np.array(row['skeletons'][joint])
            bucket_count += 1

        else:
            # divide accumulated coordinates by bucket_count to get average of data
            for joint in keep_joints:
                assert bucket_count > 0
                curr_record['skeletons'][joint] /= bucket_count

            # append curr_record to records
            records.append(curr_record)

            # update curr_bucket, create new curr_record, and reset bucket_count
            curr_bucket += delta_t
            curr_record = {
                'timestamp': curr_bucket,
                'skeletons': {
                    joint: np.zeros(3) for joint in keep_joints
                }
            }
            bucket_count = 0

    return pd.DataFrame.from_records(records)

def preprocess_pose_data(pose_fp, user_id, from_records=False):
    '''
    1. Filter out skeleton data by user_id and keep_joints
    2. Bucketize time serialized data into discrete buckets for alignment
    (see respective functions for more detailed descriptions)
    '''
    if not from_records: # HOTFIX
        pose_df = pd.read_json(pose_fp, lines=True, convert_dates=False)
        pose_df['skeletons'] = pose_df['skeletons'].apply(filter_skeleton, args=[user_id])
        pose_df = bucketize_pose_data(pose_df)
    else:
        pose_df = pd.DataFrame.from_records(pose_fp)

        # only keep joints that are measured in every frame from the records
        # HOTFIX
        keep_joints = KEEP_JOINTS
        for skeleton in pose_df['skeletons']:
            keep_joints = keep_joints & set(key for key in skeleton[0].keys())
        pose_df['skeletons'] = pose_df['skeletons'].apply(filter_skeleton, args=[user_id, keep_joints])
        pose_df = bucketize_pose_data_measurementless(pose_df, keep_joints=keep_joints)
    return pose_df



### force data processing ###

def filter_reading(reading):
    '''
    Filter the force reading data to contain only the float measurement.
    E.g., convert the original format of "Reading: 0.4 lbs" to "0.4" as a float
    '''
    pattern = 'Reading: (-?\d+(\.\d+)?) lbs'
    match = re.match(pattern, reading)
    return float(match.group(1))

def bucketize_force_data(force_df, delta_t=DELTA_T):
    '''
    Re-discretize the force data into more regular time buckets for the purpose
    of easier temporal alignment with pose data, text data, etc.

    Current strategy: use a fixed time interval between time buckets, with each
    bucket containing the averaged data of each timestamp in the range
    [bucket_t, bucket_t + delta_t) for bucket start time bucket_t. Averaged
    measurements are defined by keep_measurements
    '''
    start_t = force_df['timestamp'][0] // delta_t * delta_t
    curr_bucket = start_t

    records = []

    curr_record = {
        'timestamp': curr_bucket,
        'reading': 0.0
    }

    bucket_count = 0
    for ix, row in force_df.iterrows():
        if row['timestamp'] < curr_bucket + delta_t:
            # sum up force readings within time bucket
            curr_record['reading'] += row['reading']
            bucket_count += 1

        else:
            # divide accumulated coordinates by bucket_count to get average of data
            curr_record['reading'] /= bucket_count

            # append curr_record to records
            records.append(curr_record)

            # update curr_bucket, create new curr_record, and reset bucket_count
            curr_bucket += delta_t
            curr_record = {
                'timestamp': curr_bucket,
                'reading': 0.0
            }
            bucket_count = 0

    return pd.DataFrame.from_records(records)

def preprocess_force_data(force_fp):
    '''
    1. Filter force readings to only contain numeric measurement
    2. Bucketize time serialized data into discrete buckets for alignment
    (see respective functions for more detailed descriptions)
    '''
    force_df = pd.read_csv(force_fp)
    force_df['reading'] = force_df['reading'].apply(filter_reading)
    force_df = bucketize_force_data(force_df)
    return force_df

### data alignment ###

def align_data(text_df, pose_df, force_df, from_records=False):
    # make feature tensors the same length
    # HOTFIX
    # t_start = max(pose_df.iloc[0]['timestamp'], force_df.iloc[0]['timestamp'])
    # t_end = min(pose_df.iloc[-1]['timestamp'], force_df.iloc[-1]['timestamp'])
    # pose_df = pose_df.iloc[
    #     pose_df.index[pose_df['timestamp'] == t_start][0] : pose_df.index[pose_df['timestamp'] == t_end][0]
    # ]
    # force_df = force_df.iloc[
    #     force_df.index[force_df['timestamp'] == t_start][0] : force_df.index[force_df['timestamp'] == t_end][0]
    # ]

    # convert feature dataframes into tensors
    if not from_records: # HOTFIX
        pose_tensor = torch.tensor(
            [
                [value for joint in skeleton for measurement in skeleton[joint] for value in skeleton[joint][measurement]]
                for skeleton in pose_df['skeletons']
            ]
        )
    else:
        pose_tensor = torch.tensor(
            [
                [measurement for joint in skeleton for measurement in skeleton[joint]]
                for skeleton in pose_df['skeletons']
            ]
        )
    force_tensor = torch.tensor(force_df['reading']).reshape((-1, 1))
    text_tensor = torch.tensor(text_df['text']).reshape((-1, 1))
    # pad text_df as necessary to match length of pose and force df
    # TODO: replace padding with silence token, current 0 = [PAD] (is this ok? better to have specific silence token?)
    # text_tensor = torch.cat((text_tensor, torch.zeros(pose_tensor.shape[0] - text_tensor.shape[0], 1)), dim=0).long() # HOTFIX

    # !!! feature list of all feature tensors, add all used features here !!!
    feature_tensors = [
        # text_tensor,
        pose_tensor,
        # force_tensor,
    ]

    # check that all dataframes are the same length
    assert len(set([tensor.shape[0] for tensor in feature_tensors])) == 1, f'feature DataFrames have different lengths: {[len(tensor) for tensor in feature_tensors]}'

    # arrange data in a single pytorch tensor
    # columns represent various features (pose data, joint data, etc), rows represent time sequential data
    num_features = sum(tensor.shape[1] for tensor in feature_tensors)
    full_tensor = torch.cat(feature_tensors, dim=1)
    assert full_tensor.shape[1] == num_features

    # # reshape for batch dim first?
    # full_tensor = full_tensor.unsqueeze(0)

    # TODO: add positional embedding
    return full_tensor, text_tensor

### dataset ###


### preprocessing pipeline ###

def preprocess_pipeline(audio_fp_list, pose_fp_list, force_fp_list, tokenizer, from_records=False):
    # check that there are the same number of audio, pose, and force data files
    # assert len(set((len(audio_fp_list), len(pose_fp_list), len(force_fp_list)))) == 1 # HOTFIX

    full_feature_tensor = torch.tensor([])
    full_text_tensor = torch.tensor([], dtype=torch.int64)
    if not from_records:
        for audio_fp, pose_fp, force_fp in zip(
            audio_fp_list, pose_fp_list, force_fp_list
        ):
            text_df = preprocess_text_data(audio_fp, tokenizer)
            pose_df = preprocess_pose_data(pose_fp, user_id=1)
            force_df = preprocess_force_data(force_fp)

            aligned_tensor, text_tensor = align_data(text_df, pose_df, force_df)
            full_feature_tensor = torch.cat((full_feature_tensor, aligned_tensor), dim=0)
            full_text_tensor = torch.cat((full_text_tensor, text_tensor), dim=0)
    else: # from dataframe
        # HOTFIX, clean this up 
        text_df = preprocess_text_data(audio_fp_list, tokenizer)
        pose_df = preprocess_pose_data(pose_fp_list, 0, from_records=True) 
        force_df = preprocess_force_data(force_fp_list)
        aligned_tensor, text_tensor = align_data(text_df, pose_df, force_df, from_records=True)
        full_feature_tensor = torch.cat((full_feature_tensor, aligned_tensor), dim=0)
        full_text_tensor = torch.cat((full_text_tensor, text_tensor), dim=0)


    # conv requires shape (B, C_in, L_in)
    # full_tensor = full_tensor.permute(0,2,1)
    full_feature_tensor = full_feature_tensor.permute(1,0)
    return full_feature_tensor, full_text_tensor





if __name__ == '__main__':
    audio_fp = 'data/lightbuzz_table_1/cut_audio.wav'
    pose_fp = f'{DATA_FP}/lightbuzz_table_1/cut_poses.jsonl'
    force_fp = f'{DATA_FP}/lightbuzz_table_1/cut_data.csv'

    audio_fp_list = [
        f'{DATA_FP}/lightbuzz_table_{i}/cut_audio.wav'
        for i in range(1,2)
    ]
    pose_fp_list = [
        f'{DATA_FP}/lightbuzz_table_{i}/cut_poses.jsonl'
        for i in range(1,2)
    ]
    force_fp_list = [
        f'{DATA_FP}/lightbuzz_table_{i}/cut_data.csv'
        for i in range(1,2)
    ]

    tokenizer = transformers.BertTokenizerFast.from_pretrained('bert-base-uncased')

    # # audio testing
    # transcription = get_transcription(audio_fp)
    # print(json.dumps(transcription, indent=2))

    # # text testing
    # text_df = bucketize_text_data(transcription)
    # # text_df = preprocess_text_data(audio_fp, tokenizer)
    # print(text_df.head())

    # # pose testing
    # pose_df = preprocess_pose_data(pose_fp, user_id=1)
    # for i in range(10):
    #     print(pose_df.iloc[i]['timestamp'])
    #     print(pose_df.iloc[i]['skeletons']['ShoulderRight']['pos2D'])

    # # force testing
    # force_df = preprocess_force_data(force_fp)
    # for i in range(10):
    #     print(force_df.iloc[i]['timestamp'])
    #     print(force_df.iloc[i]['reading'])
    #     print()

    # # align testing
    # conv_input, text_tensor = preprocess_pipeline(audio_fp, pose_fp, force_fp, tokenizer)
    # print(conv_input.shape)
    # print(text_tensor.shape)

    conv_input, text_tensor = preprocess_pipeline(audio_fp_list, pose_fp_list, force_fp_list, tokenizer)
    print(conv_input)
    print(f'{conv_input.shape=}, {text_tensor.shape=}')
    print(text_tensor.dtype)

    pass