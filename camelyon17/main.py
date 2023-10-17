import data
import os
import pytorch_lightning as pl
from argparse import ArgumentParser
from erm import ERM_X
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from utils.enums import Task, EvalStage
from vae import VAE


def make_data(args):
    batch_size = args.infer_batch_size if args.task == Task.CLASSIFY else args.batch_size
    data_train, data_val_iid, data_val_ood, data_test = data.make_data(batch_size, args.n_workers, args.n_debug_examples)
    if args.eval_stage is None:
        data_eval = None
    elif args.eval_stage == EvalStage.TRAIN:
        data_eval = data_train
    elif args.eval_stage == EvalStage.VAL_ID:
        data_eval = data_val_iid
    elif args.eval_stage == EvalStage.VAL_OOD:
        data_eval = data_val_ood
    else:
        assert args.eval_stage == EvalStage.TEST
        data_eval = data_test
    return data_train, data_val_iid, data_val_ood, data_test, data_eval


def ckpt_fpath(args, task):
    return os.path.join(args.dpath, task.value, f'version_{args.seed}', 'checkpoints', 'best.ckpt')


def make_model(args):
    is_train = args.eval_stage is None
    if args.task == Task.ERM_X:
        if is_train:
            return ERM_X(args.h_sizes, args.lr, args.weight_decay)
        else:
            return ERM_X.load_from_checkpoint(ckpt_fpath(args, args.task))
    elif args.task == Task.VAE:
        return VAE(args.task, args.z_size, args.rank, args.h_sizes, args.y_mult, args.beta, args.reg_mult, args.lr,
            args.weight_decay, args.alpha, args.lr_infer, args.n_infer_steps)
    else:
        assert args.task == Task.CLASSIFY
        return VAE.load_from_checkpoint(ckpt_fpath(args, Task.VAE), task=args.task, alpha=args.alpha,
            lr_infer=args.lr_infer, n_infer_steps=args.n_infer_steps)


def main(args):
    pl.seed_everything(args.seed)
    data_train, data_val_iid, data_val_ood, data_test, data_eval = make_data(args)
    model = make_model(args)
    if args.task == Task.ERM_X:
        if args.eval_stage is None:
            trainer = pl.Trainer(
                logger=CSVLogger(os.path.join(args.dpath, args.task.value), name='', version=args.seed),
                callbacks=[
                    EarlyStopping(monitor='val_metric', mode='max', patience=int(args.early_stop_ratio * args.n_epochs)),
                    ModelCheckpoint(monitor='val_metric', mode='max', filename='best')],
                max_epochs=args.n_epochs)
            trainer.fit(model, data_train, data_val_ood)
        else:
            trainer = pl.Trainer(logger=CSVLogger(os.path.join(args.dpath, args.task.value, args.eval_stage.value),
                name='', version=args.seed), max_epochs=1)
            trainer.test(model, data_eval)
    elif args.task == Task.VAE:
        trainer = pl.Trainer(
            logger=CSVLogger(os.path.join(args.dpath, args.task.value), name='', version=args.seed),
            callbacks=[
                EarlyStopping(monitor='val_loss', patience=int(args.early_stop_ratio * args.n_epochs)),
                ModelCheckpoint(monitor='val_loss', filename='best')],
            max_epochs=args.n_epochs)
        trainer.fit(model, data_train, data_val_iid)
    else:
        assert args.task == Task.CLASSIFY
        trainer = pl.Trainer(
            logger=CSVLogger(os.path.join(args.dpath, args.task.value, f'alpha={args.alpha}', args.eval_stage.value),
                name='', version=args.seed),
            max_epochs=1,
            inference_mode=False)
        trainer.test(model, data_eval)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dpath', type=str, default='results')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--task', type=Task, choices=list(Task), required=True)
    parser.add_argument('--eval_stage', type=EvalStage, choices=list(EvalStage))
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--infer_batch_size', type=int, default=1024)
    parser.add_argument('--n_workers', type=int, default=20)
    parser.add_argument('--n_debug_examples', type=int)
    parser.add_argument('--z_size', type=int, default=200)
    parser.add_argument('--rank', type=int, default=100)
    parser.add_argument('--h_sizes', nargs='+', type=int, default=[512, 512])
    parser.add_argument('--y_mult', type=float, default=1)
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--reg_mult', type=float, default=1e-5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--lr_infer', type=float, default=1)
    parser.add_argument('--n_infer_steps', type=int, default=200)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--early_stop_ratio', type=float, default=0.1)
    main(parser.parse_args())