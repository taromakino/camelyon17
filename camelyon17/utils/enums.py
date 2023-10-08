from enum import Enum


class Task(Enum):
    ERM_X = 'erm_x'
    ERM_ZC = 'erm_zc'
    VAE = 'vae'
    Q_Z = 'q_z'
    INFER_Z = 'infer_z'


class EvalStage(Enum):
    TRAIN = 'train'
    VAL_ID = 'val_id'
    VAL_OOD = 'val_ood'
    TEST = 'test'