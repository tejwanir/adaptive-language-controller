import json
import numpy as np
import pandas as pd
import re
import torch
import transformers
import whisper

import util
import ast


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
        # print(f'{keep_joints=}')
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
    '''
    bucketize measurementless
    '''
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
                      max_text_seq_length=MAX_TEXT_SEQ_LENGTH, max_feature_seq_length=MAX_FEATURE_SEQ_LENGTH,
                      from_records=False):
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
    if text_df is not None:
        text_tensor = torch.tensor(text_df['text'])

    if not from_records:
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

    else: # from records, entire feature input forms one phrase
        pose_tensor = torch.tensor(
            [
                [measurement for joint in skeleton for measurement in skeleton[joint]]
                for skeleton in pose_df['skeletons']
            ]
        )
        feature_tensor = torch.concat((
            pose_tensor,
            torch.zeros((max_feature_seq_length - pose_tensor.shape[0], pose_tensor.shape[1]))
        ), dim=0)
        feature_tensor = feature_tensor.unsqueeze(0)

        text_tensor = torch.tensor([]) # don't need text tensor in inference
        return feature_tensor, text_tensor

def preprocess_pipeline(phrases_data, pose_data, force_data, tokenizer,
                        max_text_seq_length=MAX_TEXT_SEQ_LENGTH, from_records=False):
    """
    For now, phrases_json should be a list of file paths to phrases.json in respective
    data directories, and pose_data/force_data should be a list of file paths too.
    """
    full_feature_tensor = torch.tensor([])
    full_text_tensor = torch.tensor([], dtype=torch.int64)

    if not from_records:
        for phrases_fp, pose_fp, force_fp in zip(phrases_data, pose_data, force_data):
            text_df, phrases_json = preprocess_text_data(phrases_fp, tokenizer, max_text_seq_length=MAX_TEXT_SEQ_LENGTH)
            pose_df = preprocess_pose_data(pose_fp, user_id=1) # HARDCODED user_id
            force_df = preprocess_force_data(force_fp)

            feature_tensor, text_tensor = align_phrase_data(text_df, pose_df, force_df, phrases_json)
            full_feature_tensor = torch.cat((full_feature_tensor, feature_tensor), dim=0)
            full_text_tensor = torch.cat((full_text_tensor, text_tensor), dim=0)

    else:
        # read from records
        user_id = 0 # HARDCODED
        text_df = None # for now, not using audio/text at inference, so don't need text_df
        pose_df = None
        force_df = None
        phrases_json = None

        if pose_data is not None:
            pose_df = preprocess_pose_data(pose_data, user_id, from_records=True)

        feature_tensor, text_tensor = align_phrase_data(text_df, pose_df, force_df, phrases_json, from_records=True)
        full_feature_tensor = torch.cat((full_feature_tensor, feature_tensor), dim=0)
        full_text_tensor = torch.cat((full_text_tensor, text_tensor), dim=0)

    return full_feature_tensor, full_text_tensor


if __name__ == '__main__':
    audio_fp = 'data/lightbuzz_table_1/cut_audio.wav'
    pose_fp = 'data/lightbuzz_table_1/cut_poses.jsonl'
    force_fp = 'data/lightbuzz_table_1/cut_data.csv'

    tokenizer = transformers.BertTokenizerFast.from_pretrained('bert-base-uncased')

    phrases_fp = 'data/lightbuzz_table_1/phrases.json'

    file_range = range(1,7)
    phrases_fp_list, pose_fp_list, force_fp_list = [], [] ,[]
    for i in file_range:
        phrases_fp_list.append(f'data/lightbuzz_table_{i}/phrases.json')
        pose_fp_list.append(f'data/lightbuzz_table_{i}/cut_poses.jsonl')
        force_fp_list.append(f'data/lightbuzz_table_{i}/cut_data.csv')

    # with open('data/lightbuzz_table_1/transcription_base.json', 'r') as f:
    #     transcription = json.load(f)
    # segments = util.parse_transcription_to_phrases(transcription)
    # print(json.dumps(segments, indent=2))

    # ### generate transcription files
    # i = 6
    # with open(f'data/lightbuzz_table_{i}/transcription_base.json', 'r') as f:
    #     transcription = json.load(f)
    # print(f'start {i}')
    # segments = util.parse_transcription_to_phrases(transcription)
    # with open(f'data/lightbuzz_table_{i}/phrases.json', 'w') as f:
    #     json.dump(segments, f, indent=2)
    # print(f'end {i}')

    # ### full pipeline testing (from list of data filepaths)
    # full_feature_tensor, text_tensor = preprocess_pipeline(
    #     phrases_fp_list, pose_fp_list, force_fp_list, tokenizer
    # )
    # print(full_feature_tensor.shape)
    # print(text_tensor.shape)



    ### from_records testing
    pose_records = [{0: {'Chest': np.array([-0.01459269,  0.0600051 ,  0.00790584]), 'ClavicleLeft': np.array([ 0.05559847,  0.00125566, -0.00702226]), 'ClavicleRight': np.array([-0.05533162,  0.0045778 ,  0.01436353]), 'ElbowLeft': np.array([ 0.21007458,  0.26003872, -0.11569858]), 'ElbowRight': np.array([-0.15069339,  0.23269396, -0.06561697]), 'Neck': np.array([0., 0., 0.]), 'ShoulderLeft': np.array([ 0.18532821,  0.00418551, -0.02340746]), 'ShoulderRight': np.array([-0.18443871,  0.01525933,  0.0478785 ]), 'WristLeft': np.array([ 0.10692689,  0.38775884, -0.28212738]), 'WristRight': np.array([-0.08911604,  0.38752392, -0.12160408])}}, {0: {'BackSkull': np.array([-0.04464875, -0.12311498,  0.01166055]), 'ClavicleLeft': np.array([ 0.05597088,  0.00115928, -0.00693435]), 'ClavicleRight': np.array([-0.05555348,  0.00465115,  0.01418541]), 'ElbowLeft': np.array([ 0.21165367,  0.26165488, -0.11676701]), 'ElbowRight': np.array([-0.15221028,  0.23417341, -0.06316858]), 'EyeLeft': np.array([-0.0271335 , -0.11333034, -0.0465484 ]), 'Head': np.array([-0.04453244, -0.09805941, -0.01333395]), 'Neck': np.array([0., 0., 0.]), 'Nose': np.array([-0.04430085, -0.0479311 , -0.06331757]), 'ShoulderLeft': np.array([ 0.18659182,  0.00384567, -0.02306368]), 'ShoulderRight': np.array([-0.18515371,  0.01549427,  0.04730901]), 'WristLeft': np.array([ 0.10782907,  0.39010332, -0.28437519]), 'WristRight': np.array([-0.09005539,  0.39031935, -0.1198295 ])}}, {0: {'BackSkull': np.array([-0.04171941, -0.121761  ,  0.00645779]), 'ClavicleLeft': np.array([ 0.0559341 ,  0.00100405, -0.00678539]), 'ClavicleRight': np.array([-0.05535732,  0.00465027,  0.01395884]), 'ElbowLeft': np.array([ 0.21184692,  0.26148179, -0.12013487]), 'ElbowRight': np.array([-0.15304637,  0.23415859, -0.05891186]), 'EyeLeft': np.array([-0.02525354, -0.111777  , -0.05141029]), 'Head': np.array([-0.04237169, -0.09688625, -0.01807622]), 'Neck': np.array([0., 0., 0.]), 'Nose': np.array([-0.04297277, -0.04717014, -0.06755108]), 'ShoulderLeft': np.array([ 0.1864877 ,  0.00330629, -0.02253287]), 'ShoulderRight': np.array([-0.18450052,  0.0154864 ,  0.04656361]), 'WristLeft': np.array([ 0.10850996,  0.38887884, -0.29072712]), 'WristRight': np.array([-0.09051277,  0.39062879, -0.116648  ])}}, {0: {'BackSkull': np.array([-0.03975932, -0.11939016,  0.00301451]), 'ClavicleLeft': np.array([ 0.0558963 ,  0.00094596, -0.00672562]), 'ClavicleRight': np.array([-0.05528559,  0.00463026,  0.01390698]), 'ElbowLeft': np.array([ 0.2118464 ,  0.26142873, -0.12227867]), 'ElbowRight': np.array([-0.15346441,  0.23419436, -0.05713037]), 'EyeLeft': np.array([-0.02366809, -0.10812642, -0.05545037]), 'Head': np.array([-0.0409172 , -0.0945934 , -0.02125675]), 'Neck': np.array([0., 0., 0.]), 'Nose': np.array([-0.04213358, -0.04506868, -0.0704287 ]), 'ShoulderLeft': np.array([ 0.18637342,  0.00310874, -0.02234252]), 'ShoulderRight': np.array([-0.18425742,  0.01541966,  0.0463798 ]), 'WristLeft': np.array([ 0.10882708,  0.38821283, -0.29458502]), 'WristRight': np.array([-0.09079855,  0.39081528, -0.11523067])}}, {0: {'BackSkull': np.array([-0.0371861 , -0.11723135, -0.00214649]), 'ClavicleLeft': np.array([ 0.0566298 ,  0.00086569, -0.00661306]), 'ClavicleRight': np.array([-0.05596663,  0.0046916 ,  0.01385451]), 'EarLeft': np.array([ 0.02070278, -0.09981012, -0.06026934]), 'ElbowLeft': np.array([ 0.21490549,  0.26523127, -0.12508261]), 'ElbowRight': np.array([-0.15653154,  0.23766479, -0.05483193]), 'EyeLeft': np.array([-0.0215898 , -0.10383404, -0.06182603]), 'Head': np.array([-0.03926118, -0.0922855 , -0.02603202]), 'Neck': np.array([0., 0., 0.]), 'Nose': np.array([-0.04171041, -0.04257381, -0.07470359]), 'ShoulderLeft': np.array([ 0.18888773,  0.00284542, -0.02200954]), 'ShoulderRight': np.array([-0.18645159,  0.01561368,  0.04618066]), 'WristLeft': np.array([ 0.1104758 ,  0.39371883, -0.29971944]), 'WristRight': np.array([-0.09296444,  0.39701188, -0.11333628])}}] * 3

    interval_records = {
        'timestamp': np.arange(0, len(pose_records)) * 0.1,
        'skeletons': pose_records,
    }

    # pose_df = preprocess_pose_data(interval_records, 0, from_records=True)
    # feature_tensor, text_tensor = align_phrase_data(None, pose_df, None, None, from_records=True)

    feature_tensor, text_tensor = preprocess_pipeline(None, interval_records, None, tokenizer, from_records=True)
    print(f'{feature_tensor=}')
    print(f'{feature_tensor.shape=}')