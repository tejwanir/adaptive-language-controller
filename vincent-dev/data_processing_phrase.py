import json
import numpy as np
import pandas as pd
import re
import torch
import transformers
import whisper

import util


### global constants ###
DELTA_T = 0.2
# as of 5/20/24, server only outputs WristLeft pos3D
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
MAX_TEXT_SEQ_LENGTH = 32
MAX_FEATURE_SEQ_LENGTH = 64

### text processing ###
def preprocess_text_data(phrases_fp, tokenizer, delta_t=DELTA_T,
                         max_text_seq_length=MAX_TEXT_SEQ_LENGTH):
    '''
    phrases_fp should be the file path to phrases.json
    '''
    with open(phrases_fp, 'r') as f:
        phrases_json = json.load(f)
    precision = 1 # HARDCODED to match DELTA_T = 0.2 with precision
    text_df = pd.DataFrame([phrase['phrase'] for phrase in phrases_json], columns=['text'])
    # tokenize words; text should already be normalized from phrase generation
    text_df['text'] = text_df['text'].apply(
        lambda text: tokenizer.encode(text, padding='max_length', max_length=max_text_seq_length)
    )
    # bucketize phrase times to align evenly with delta_t
    for phrase in phrases_json:
        phrase['start_time'] = round((phrase['start_time'] // delta_t) * delta_t, precision)
        phrase['end_time'] = round((phrase['end_time'] // delta_t) * delta_t, precision)

    return text_df, phrases_json


### pose processing ###
def filter_skeleton(skeletons_list, user_id, keep_joints=KEEP_JOINTS):
    '''
    Filter the pose data to contain only the user specified by `user_id`.
    Filter `user_id` skeleton to contain only desired joints (defined by keep_joints).
    Return dicitonary object of remaining pose data.

    pose_df['skeletons'] is a list of json objects, with each element
    corresponding to a different "user" from the original video
    '''
    # TODO: filter by confidence threshold too?
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
    float_precision = 1 # HARDCODED for DELTA_T=0.2
    start_t = round(pose_df['timestamp'][0] // delta_t * delta_t, float_precision)
    curr_bucket = start_t

    records = []

    curr_record = {
        'timestamp': round(curr_bucket - start_t, float_precision), # timestamps start at time 0
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
                'timestamp': round(curr_bucket - start_t, float_precision), # timestamps start at time 0
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
#     '''
#     bucketize measurementless
#     '''
#     start_t = pose_df['timestamp'][0] // delta_t * delta_t
#     curr_bucket = start_t

#     records = []

#     curr_record = {
#         'timestamp': curr_bucket,
#         'skeletons': {
#             joint: np.zeros(3) for joint in keep_joints
#         }
#     }

#     bucket_count = 0
#     for ix, row in pose_df.iterrows():
#         if row['timestamp'] < curr_bucket + delta_t:
#             # sum up data for each measurement, for each joint
#             for joint in keep_joints:
#                 curr_record['skeletons'][joint] += np.array(row['skeletons'][joint])
#             bucket_count += 1

#         else:
#             # divide accumulated coordinates by bucket_count to get average of data
#             for joint in keep_joints:
#                 assert bucket_count > 0
#                 curr_record['skeletons'][joint] /= bucket_count

#             # append curr_record to records
#             records.append(curr_record)

#             # update curr_bucket, create new curr_record, and reset bucket_count
#             curr_bucket += delta_t
#             curr_record = {
#                 'timestamp': curr_bucket,
#                 'skeletons': {
#                     joint: np.zeros(3) for joint in keep_joints
#                 }
#             }
#             bucket_count = 0

#     return pd.DataFrame.from_records(records)
    pass

def preprocess_pose_data(pose_data, user_id, from_records=False):
    '''
    If from_records=False, pose_data should be a valid filepath containing pose data
    If from_records=True, pose_data should be a dict of records (see formatting in preprocess_pipeline)

    1. Filter out skeleton data by user_id and keep_joints
    2. Bucketize time serialized data into discrete buckets for alignment
    (see respective functions for more detailed descriptions)
    '''
    if not from_records: # HOTFIX
        pose_df = pd.read_json(pose_data, lines=True, convert_dates=False)
        pose_df['skeletons'] = pose_df['skeletons'].apply(filter_skeleton, args=[user_id])
        pose_df = bucketize_pose_data(pose_df)
    else:
        pose_df = pd.DataFrame.from_records(pose_data)

        # only keep joints that are measured in every frame from the records
        # HOTFIX
        keep_joints = KEEP_JOINTS
        for skeleton in pose_df['skeletons']:
            keep_joints = keep_joints & set(key for key in skeleton[0].keys())
        pose_df['skeletons'] = pose_df['skeletons'].apply(filter_skeleton, args=[user_id, keep_joints])
        pose_df = bucketize_pose_data_measurementless(pose_df, keep_joints=keep_joints)
    return pose_df


### force processing ###
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
    float_precision = 1 # HARDCODED for DELTA_T=0.2
    start_t = round(force_df['timestamp'][0] // delta_t * delta_t, float_precision)
    curr_bucket = start_t

    records = []

    curr_record = {
        'timestamp': round(curr_bucket - start_t, float_precision), # timestamps start at time 0
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
                'timestamp': round(curr_bucket - start_t, float_precision), # timestamps start at time 0
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


### alignment ###
def align_phrase_data(text_df, pose_df, force_df, phrases_json,
                      max_text_seq_length=MAX_TEXT_SEQ_LENGTH, max_feature_seq_length=MAX_FEATURE_SEQ_LENGTH):
    # make feature tensors the same length
    if pose_df is not None and force_df is not None:
        t_start = max(pose_df.iloc[0]['timestamp'], force_df.iloc[0]['timestamp'])
        t_end = min(pose_df.iloc[-1]['timestamp'], force_df.iloc[-1]['timestamp'])
        pose_df = pose_df.iloc[
            pose_df.index[pose_df['timestamp'] == t_start][0] : pose_df.index[pose_df['timestamp'] == t_end][0]
        ]
        force_df = force_df.iloc[
            force_df.index[force_df['timestamp'] == t_start][0] : force_df.index[force_df['timestamp'] == t_end][0]
        ]

    # convert text to tensor (text should already be padded to MAX_TEXT_LENGTH)
    text_tensor = torch.tensor(text_df['text'])

    full_feature_tensor = torch.tensor([])
    # for both pose and force tensors:
    # read phrase json for start/end times
    for ix, phrase in enumerate(phrases_json):
        # extract features from current phrase start time to next phrase start time
        curr_phrase_start_time = phrase['start_time']
        if ix < len(phrases_json) - 1:
            next_phrase_start_time = phrases_json[ix+1]['start_time']
        else:
            next_phrase_start_time = np.inf
        # extract measurements corresponding to that time range
        phrase_pose_df = pose_df[
            (curr_phrase_start_time <= pose_df['timestamp']) &
            (pose_df['timestamp'] < next_phrase_start_time)
        ]
        phrase_force_df = force_df[
            (curr_phrase_start_time <= force_df['timestamp']) &
            (force_df['timestamp'] < next_phrase_start_time)
        ]
        assert (len(phrase_pose_df) == len(phrase_force_df))

        # convert to tensor, concatenate pose and force tensors
        phrase_pose_tensor = torch.tensor(
            [
                [value for joint in skeleton for measurement in skeleton[joint] for value in skeleton[joint][measurement]]
                for skeleton in phrase_pose_df['skeletons']
            ]
        )
        phrase_force_tensor = torch.tensor(phrase_force_df['reading'].values).view(-1, 1)
        phrase_feature_tensor = torch.concat((phrase_pose_tensor, phrase_force_tensor), dim=1)[:max_feature_seq_length]
        # pad to max feature length
        assert phrase_feature_tensor.shape[0] <= max_feature_seq_length
        phrase_feature_tensor = torch.concat((
            phrase_feature_tensor,
            torch.zeros((max_feature_seq_length - phrase_feature_tensor.shape[0], phrase_feature_tensor.shape[1]))
        ), dim=0)
        # concatenate to full feature tensor along dim 0
        phrase_feature_tensor = phrase_feature_tensor.unsqueeze(0)
        full_feature_tensor = torch.concat((full_feature_tensor, phrase_feature_tensor))

    # output: text_tensor (num_phrases x max_text_len)
    # output: aligned_tensor (num_phrases x max_feature_len x num_features)
    return full_feature_tensor, text_tensor

def preprocess_pipeline(phrases_data, pose_data, force_data, tokenizer,
                        max_text_seq_length=MAX_TEXT_SEQ_LENGTH):
    """
    For now, phrases_json should be a list of file paths to phrases.json in respective
    data directories, and pose_data/force_data should be a list of file paths too.
    """
    full_feature_tensor = torch.tensor([])
    full_text_tensor = torch.tensor([], dtype=torch.int64)

    for phrases_fp, pose_fp, force_fp in zip(phrases_data, pose_data, force_data):
        text_df, phrases_json = preprocess_text_data(phrases_fp, tokenizer, max_text_seq_length=MAX_TEXT_SEQ_LENGTH)
        pose_df = preprocess_pose_data(pose_fp, user_id=1) # HARDCODED user_id
        force_df = preprocess_force_data(force_fp)

        feature_tensor, text_tensor = align_phrase_data(text_df, pose_df, force_df, phrases_json)
        full_feature_tensor = torch.cat((full_feature_tensor, feature_tensor), dim=0)
        full_text_tensor = torch.cat((full_text_tensor, text_tensor), dim=0)
    return full_feature_tensor, full_text_tensor


if __name__ == '__main__':
    audio_fp = 'data/lightbuzz_table_1/cut_audio.wav'
    pose_fp = 'data/lightbuzz_table_1/cut_poses.jsonl'
    force_fp = 'data/lightbuzz_table_1/cut_data.csv'

    tokenizer = transformers.BertTokenizerFast.from_pretrained('bert-base-uncased')

    phrases_fp = 'data/lightbuzz_table_1/phrases.json'

    # with open('data/lightbuzz_table_1/transcription_base.json', 'r') as f:
    #     transcription = json.load(f)
    # segments = util.parse_transcription_to_phrases(transcription)
    # print(json.dumps(segments, indent=2))

    file_range = range(1,7)
    phrases_fp_list, pose_fp_list, force_fp_list = [], [] ,[]
    for i in file_range:
        phrases_fp_list.append(f'data/lightbuzz_table_{i}/phrases.json')
        pose_fp_list.append(f'data/lightbuzz_table_{i}/cut_poses.jsonl')
        force_fp_list.append(f'data/lightbuzz_table_{i}/cut_data.csv')

    full_feature_tensor, text_tensor = preprocess_pipeline(
        phrases_fp_list, pose_fp_list, force_fp_list, tokenizer
    )
    print(full_feature_tensor.shape)
    print(text_tensor.shape)

    # print(json.dumps(tokenizer.batch_decode(text_tensor), indent=2))

    # i = 6
    # with open(f'data/lightbuzz_table_{i}/transcription_base.json', 'r') as f:
    #     transcription = json.load(f)
    # print(f'start {i}')
    # segments = util.parse_transcription_to_phrases(transcription)
    # with open(f'data/lightbuzz_table_{i}/phrases.json', 'w') as f:
    #     json.dump(segments, f, indent=2)
    # print(f'end {i}')