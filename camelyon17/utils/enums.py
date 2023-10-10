from enum import Enum


class Task(Enum):
    ERM_X = 'erm_x'
    VAE = 'vae'
    Q_Z = 'q_z'
    CLASSIFY = 'classify'


class EvalStage(Enum):
    TRAIN = 'train'
    VAL_ID = 'val_id'
    VAL_OOD = 'val_ood'
    TEST = 'test'