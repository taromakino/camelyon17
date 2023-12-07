from enum import Enum


class Task(Enum):
    ERM = 'erm'
    VAE = 'vae'
    CLASSIFY = 'classify'


class EvalStage(Enum):
    TRAIN = 'train'
    VAL_ID = 'val_id'
    VAL_OOD = 'val_ood'
    TEST = 'test'