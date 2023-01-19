import pickle
from hashlib import sha256


def hex_dict(dct: dict):
    return sha256(pickle.dumps(dct)).hexdigest()
