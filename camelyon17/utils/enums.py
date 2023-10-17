from enum import Enum


class Task(Enum):
    ERM_X = 'erm_x'
    VAE = 'vae'
    CLASSIFY = 'classify'


class EvalStage(Enum):
    TRAIN = 'train'
    VAL_ID = 'val_id'
    VAL_OOD = 'val_ood'
    TEST = 'test'