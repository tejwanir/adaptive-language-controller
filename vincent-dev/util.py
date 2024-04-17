import json
import numpy as np

# from https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
# for json dumping np arrays

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)