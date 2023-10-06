import os
import pickle


def save_file(obj, fpath):
    dpath = os.path.dirname(fpath)
    os.makedirs(dpath, exist_ok=True)
    with open(fpath, 'wb') as file:
        pickle.dump(obj, file)


def load_file(fpath):
    with open(fpath, 'rb') as file:
        return pickle.load(file)


def write(fpath, text):
    with open(fpath, 'a+') as f:
        f.write(text + '\n')