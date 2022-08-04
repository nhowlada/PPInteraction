import argparse
import concurrent.futures

import os
import sys
import traceback
from collections import defaultdict
from datetime import datetime

from importlib_metadata import metadata
from commons.logger import Logger

from datasets.balancedbatchsampler import BalancedBatchSampler
from datasets.ppi_dataset import PPIDataset
from datasets.transforms import *
from commons.utils import seed_all, get_random_indices, log, padded_permuted_collate

import yaml
#from datasets.custom_collate import *  # do not remove
from models import *  # do not remove
from torch.nn import *  # do not remove
from torch.optim import *  # do not remove
from commons.losses import *  # do not remove
from torch.optim.lr_scheduler import *  # do not remove
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Subset

from trainer.metrics import MAE, Bal_Accuracy, Accuracy, MCC, F1_Score, Precision, Recall
from trainer.trainer import Trainer

# turn on for debugging for C code like Segmentation Faults
import faulthandler

faulthandler.enable()


def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=argparse.FileType(mode='r'), default='configs_clean/linear_attention.yml')
    p.add_argument('--logdir', type=str, default='runs', help='tensorboard logdirectory')
    p.add_argument('--num_epochs', type=int, default=2500, help='number of times to iterate through all samples')
    p.add_argument('--batch_size', type=int, default=1024, help='samples that will be processed in parallel')
    p.add_argument('--multithreaded_seeds', type=list, default=[],
                   help='if this is non empty, multiple threads will be started, training the same model but with the different seeds')
    p.add_argument('--tensorboard_functions', default=[], help='choices of the TENSORBOARD_FUNCTIONS in utils')
    p.add_argument('--checkpoint', type=str, help='path to directory that contains a checkpoint to continue training')
    p.add_argument('--collate_params', type=dict, default={},
                   help='parameters with keywords of the chosen collate function')
    p.add_argument('--models_to_save', type=list, default=[],
                   help='specify after which epochs to remember the best model')
    p.add_argument('--metrics', default=[], help='tensorboard metrics [mae, mae_denormalized, qm9_properties ...]')
    p.add_argument('--sampler_parameters', type=dict, help='dictionary of sampler parameters')

    return p.parse_args()


def get_trainer(args, model, data, device, metrics, run_dir, sampler=None):
    if args.trainer == None:
        trainer = Trainer
    elif args.trainer == 'binding':
        trainer = Trainer
    # loss_func=globals()[args.loss_func](**args.loss_params)
    return trainer(model=model, args=args, metrics=metrics, main_metric=args.main_metric,
                   main_metric_goal=args.main_metric_goal, optim=globals()[args.optimizer],
                   loss_func=globals()[args.loss_func], device=device, scheduler_step_per_batch=args.scheduler_step_per_batch,
                   run_dir=run_dir, weight=data.class_weights, sampler=sampler)


def load_model(args, data_sample, device, **kwargs):
    model = globals()[args.model_type](device=device,
                                           embeddings_dim=data_sample[0][0].shape[-1],
                                           **args.model_parameters, **kwargs)
    return model

def load_ensemble_model(args, modelA, modelB, modelC, **kwargs):
    model = globals()['Ensemble'](modelA=modelA, modelB=modelB, modelC=modelC, **kwargs)
    return model


def train_wrapper(args):
    mp = args.model_parameters
    lp = args.loss_params
    if args.checkpoint:
        run_dir = os.path.dirname(args.checkpoint)
    else:
        if args.trainer == 'torsion':
            run_dir= f'{args.logdir}/{os.path.splitext(os.path.basename(args.config))[0]}_{args.experiment_name}_bs{args.batch_size}_numtrain{args.num_train}_{start_time}'
        else:
            run_dir = f'{args.logdir}/{os.path.splitext(os.path.basename(args.config))[0]}_{args.experiment_name}_bs{args.batch_size}_otL{lp["ot_loss_weight"]}_numtrain{args.num_train}_{start_time}'
    if not os.path.exists(run_dir):
        os.mkdir(run_dir)
    sys.stdout = Logger(logpath=os.path.join(run_dir, f'log.log'), syspart=sys.stdout)
    sys.stderr = Logger(logpath=os.path.join(run_dir, f'log.log'), syspart=sys.stderr)
    return train(args, run_dir)



def train(args, run_dir):
    seed_all(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == 'cuda' else "cpu")
    transform = transforms.Compose([ToTensor()])
    metrics_dict = {'bal_accuracy': Bal_Accuracy(),
                    'accuracy': Accuracy(),
                    'mcc': MCC(),
                    'f1_score': F1_Score(),
                    'precision': Precision(),
                    'recall': Recall(),
                    'mae': MAE(),
                    }

    train_data = PPIDataset(device=device, embeddings_path=args.embeddings,annotations_path=args.train_annotations, remapped_sequences=args.remapped_sequences, transform=transform, **args.dataset_params)
    val_data = PPIDataset(device=device, embeddings_path=args.embeddings,annotations_path=args.val_annotations, remapped_sequences=args.remapped_sequences, transform=transform, **args.dataset_params)

    temp = train_data.class_weights
    if args.num_train != None:
        train_data = Subset(train_data, get_random_indices(len(train_data))[:args.num_train])
        train_data.class_weights = temp
        temp=''
    if args.num_val != None:
        val_data = Subset(val_data, get_random_indices(len(val_data))[:args.num_val])

    log('train size: ', len(train_data))
    log('val size: ', len(val_data))

    model = load_model(args, data_sample=train_data[0], device=device)
    log('trainable params in model: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
    collate_function = globals()[args.collate_function] if args.collate_params == {} else globals()[
        args.collate_function](**args.collate_params)
    if len(train_data[0][0].shape) == 2:  # if we have per residue embeddings they have an additional length dim
        collate_function = padded_permuted_collate
    else:  # if we have reduced sequence wise embeddings use the default collate function by passing None
        collate_function = None
    if args.train_sampler != None:
        # sampler = globals()[args.train_sampler](data_source=train_data, batch_size=args.batch_size)
        sampler = globals()[args.train_sampler](data_source=train_data, batch_size=args.batch_size)
        train_loader = DataLoader(train_data, batch_sampler=sampler, collate_fn=collate_function,
                                  pin_memory=args.pin_memory, num_workers=args.num_workers)
    else:
        sampler = None
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_function,
                                  pin_memory=args.pin_memory, num_workers=args.num_workers)
    
    if args.train_sampler != None:
        val_sampler = globals()[args.train_sampler](data_source=val_data, batch_size=args.batch_size)
        val_loader = DataLoader(val_data, batch_sampler=val_sampler, collate_fn=collate_function,
                                  pin_memory=args.pin_memory, num_workers=args.num_workers)
    else:
        val_sampler = None
        val_loader = DataLoader(val_data, batch_size=args.batch_size, collate_fn=collate_function,
                            pin_memory=args.pin_memory, num_workers=args.num_workers)
    # train_features, train_labels, metadata = next(iter(train_loader))

    metrics = {metric: metrics_dict[metric] for metric in args.metrics}
    trainer = get_trainer(args=args, model=model, data=train_data, device=device, metrics=metrics, run_dir=run_dir,
                          sampler=sampler)
    val_metrics, _, _ = trainer.train(train_loader, val_loader)
    if args.eval_on_test:
        test_data = PPIDataset(device=device, embeddings_path=args.embeddings,annotations_path=args.test_annotations, remapped_sequences=args.remapped_sequences, transform=transform, **args.dataset_params)
        
        if args.train_sampler != None:
            test_sampler = globals()[args.train_sampler](data_source=test_data, batch_size=args.batch_size)
            test_loader = DataLoader(test_data, batch_sampler=test_sampler, collate_fn=collate_function,
                                    pin_memory=args.pin_memory, num_workers=args.num_workers)
        else:
            test_sampler = None
            test_loader = DataLoader(test_data, batch_size=args.batch_size, collate_fn=collate_function,
                                 pin_memory=args.pin_memory, num_workers=args.num_workers)
        log('test size: ', len(test_data))
        test_metrics, _, _ = trainer.evaluation(test_loader, data_split='test')
        return val_metrics, test_metrics, trainer.writer.log_dir
    return val_metrics


def get_arguments():
    args = parse_arguments()
    if args.config:
        config_dict = yaml.load(args.config, Loader=yaml.FullLoader)
        arg_dict = args.__dict__
        for key, value in config_dict.items():
            if isinstance(value, list):
                for v in value:
                    arg_dict[key].append(v)
            else:
                arg_dict[key] = value
        args.config = args.config.name
    else:
        config_dict = {}

    if args.checkpoint:  # overwrite args with args from checkpoint except for the args that were contained in the config file
        arg_dict = args.__dict__
        with open(os.path.join(os.path.dirname(args.checkpoint), 'train_arguments.yaml'), 'r') as arg_file:
            checkpoint_dict = yaml.load(arg_file, Loader=yaml.FullLoader)
        for key, value in checkpoint_dict.items():
            if key not in config_dict.keys():
                if isinstance(value, list):
                    for v in value:
                        arg_dict[key].append(v)
                else:
                    arg_dict[key] = value

    return args


def main_function():
    args = get_arguments()

    if args.multithreaded_seeds != []:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for seed in args.multithreaded_seeds:
                args_copy = get_arguments()
                args_copy.seed = seed
                futures.append(executor.submit(train_wrapper, args_copy))
            results = [f.result() for f in
                       futures]  # list of tuples of dictionaries with the validation results first and the test results second
        all_val_metrics = defaultdict(list)
        all_test_metrics = defaultdict(list)
        log_dirs = []
        for result in results:
            val_metrics, test_metrics, log_dir = result
            log_dirs.append(log_dir)
            for key in val_metrics.keys():
                all_val_metrics[key].append(val_metrics[key])
                all_test_metrics[key].append(test_metrics[key])
        files = [open(os.path.join(dir, 'multiple_seed_validation_statistics.txt'), 'w') for dir in log_dirs]
        print('Validation results:')
        for key, value in all_val_metrics.items():
            metric = np.array(value)
            for file in files:
                file.write(f'\n{key:}\n')
                file.write(f'mean: {metric.mean()}\n')
                file.write(f'stddev: {metric.std()}\n')
                file.write(f'stderr: {metric.std() / np.sqrt(len(metric))}\n')
                file.write(f'values: {value}\n')
            print(f'\n{key}:')
            print(f'mean: {metric.mean()}')
            print(f'stddev: {metric.std()}')
            print(f'stderr: {metric.std() / np.sqrt(len(metric))}')
            print(f'values: {value}')
        for file in files:
            file.close()
        files = [open(os.path.join(dir, 'multiple_seed_test_statistics.txt'), 'w') for dir in log_dirs]
        print('Test results:')
        for key, value in all_test_metrics.items():
            metric = np.array(value)
            for file in files:
                file.write(f'\n{key:}\n')
                file.write(f'mean: {metric.mean()}\n')
                file.write(f'stddev: {metric.std()}\n')
                file.write(f'stderr: {metric.std() / np.sqrt(len(metric))}\n')
                file.write(f'values: {value}\n')
            print(f'\n{key}:')
            print(f'mean: {metric.mean()}')
            print(f'stddev: {metric.std()}')
            print(f'stderr: {metric.std() / np.sqrt(len(metric))}')
            print(f'values: {value}')
        for file in files:
            file.close()
    else:
        train_wrapper(args)


if __name__ == '__main__':
    start_time = datetime.now().strftime('date%d-%m_time%H-%M-%S.%f')
    if not os.path.exists('logs'):
        os.mkdir('logs')
    with open(os.path.join('logs', f'{start_time}.log'), "w") as file:
        try:
            main_function()
        except Exception as e:
            traceback.print_exc(file=file)
            raise
