import sys, os, yaml, argparse
from unittest import result
from tqdm import tqdm
from copy import copy, deepcopy
from sklearn.metrics import matthews_corrcoef, confusion_matrix, f1_score
from sklearn import metrics

from train import load_model
from commons.utils import seed_all, read_strings_from_txt, padded_permuted_collate, move_to_device
from commons.logger import Logger
# now from datasets.custom_collate import *  # do not remove
from datasets.ppi_dataset import PPIDataset
from datasets.transforms import *
from models import *  # do not remove

from torchvision.transforms import transforms
from torch.nn import *  # do not remove
from torch.optim import *  # do not remove
from commons.losses import *  # do not remove
from torch.optim.lr_scheduler import *  # do not remove
from torch.utils.data import DataLoader
# turn on for debugging C code like Segmentation Faults
import faulthandler

faulthandler.enable()


def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=argparse.FileType(mode='r'), default='configs_clean/inference.yml')
    p.add_argument('--checkpoint', type=str, help='path to .pt file in a checkpoint directory')
    p.add_argument('--output_directory', type=str, default=None, help='path where to put the predicted results')
    p.add_argument('--run_corrections', type=bool, default=False,
                   help='whether or not to run the fast point cloud ligand fitting')
    p.add_argument('--run_dirs', type=list, default=[], help='path directory with saved runs')
    p.add_argument('--fine_tune_dirs', type=list, default=[], help='path directory with saved finetuning runs')
    p.add_argument('--inference_path', type=str, help='path to some pdb files for which you want to run inference')
    p.add_argument('--experiment_name', type=str, help='name that will be added to the runs folder output')
    p.add_argument('--logdir', type=str, default='runs', help='tensorboard logdirectory')
    p.add_argument('--num_epochs', type=int, default=2500, help='number of times to iterate through all samples')
    p.add_argument('--batch_size', type=int, default=1024, help='samples that will be processed in parallel')
    p.add_argument('--patience', type=int, default=20, help='stop training after no improvement in this many epochs')
    p.add_argument('--minimum_epochs', type=int, default=0, help='minimum numer of epochs to run')
    p.add_argument('--dataset_params', type=dict, default={},
                   help='parameters with keywords of the dataset')
    p.add_argument('--num_train', type=int, default=-1, help='n samples of the model samples to use for train')
    p.add_argument('--num_val', type=int, default=None, help='n samples of the model samples to use for validation')
    p.add_argument('--seed', type=int, default=1, help='seed for reproducibility')
    p.add_argument('--multithreaded_seeds', type=list, default=[],
                   help='if this is non empty, multiple threads will be started, training the same model but with the different seeds')
    p.add_argument('--seed_data', type=int, default=1, help='if you want to use a different seed for the datasplit')
    p.add_argument('--loss_func', type=str, default='MSELoss', help='Class name of torch.nn like [MSELoss, L1Loss]')
    p.add_argument('--loss_params', type=dict, default={}, help='parameters with keywords of the chosen loss function')
    p.add_argument('--optimizer', type=str, default='Adam', help='Class name of torch.optim like [Adam, SGD, AdamW]')
    p.add_argument('--optimizer_params', type=dict, help='parameters with keywords of the chosen optimizer like lr')
    p.add_argument('--clip_grad', type=float, default=None, help='clip gradients if magnitude is greater')
    p.add_argument('--lr_scheduler', type=str,
                   help='Class name of torch.optim.lr_scheduler like [CosineAnnealingLR, ExponentialLR, LambdaLR]')
    p.add_argument('--lr_scheduler_params', type=dict, help='parameters with keywords of the chosen lr_scheduler')
    p.add_argument('--scheduler_step_per_batch', default=True, type=bool,
                   help='step every batch if true step every epoch otherwise')
    p.add_argument('--log_iterations', type=int, default=-1,
                   help='log every log_iterations iterations (-1 for only logging after each epoch)')
    p.add_argument('--expensive_log_iterations', type=int, default=100,
                   help='frequency with which to do expensive logging operations')
    p.add_argument('--eval_per_epochs', type=int, default=0,
                   help='frequency with which to do run the function run_eval_per_epoch that can do some expensive calculations on the val set or sth like that. If this is zero, then the function will never be called')
    p.add_argument('--metrics', default=[], help='tensorboard metrics [mae, mae_denormalized, qm9_properties ...]')
    p.add_argument('--main_metric', default='loss', help='for early stopping etc.')
    p.add_argument('--main_metric_goal', type=str, default='min', help='controls early stopping. [max, min]')
    p.add_argument('--val_per_batch', type=bool, default=True,
                   help='run evaluation every batch and then average over the eval results. When running the molhiv benchmark for example, this needs to be Fale because we need to evaluate on all val data at once since the metric is rocauc')
    p.add_argument('--tensorboard_functions', default=[], help='choices of the TENSORBOARD_FUNCTIONS in utils')
    p.add_argument('--num_epochs_local_only', type=int, default=1,
                   help='when training with OptimalTransportTrainer, this specifies for how many epochs only the local predictions will get a loss')

    p.add_argument('--collate_function', default='graph_collate', help='the collate function to use for DataLoader')
    p.add_argument('--collate_params', type=dict, default={},
                   help='parameters with keywords of the chosen collate function')
    p.add_argument('--device', type=str, default='cuda', help='What device to train on: cuda or cpu')

    p.add_argument('--models_to_save', type=list, default=[],
                   help='specify after which epochs to remember the best model')

    p.add_argument('--model_type', type=str, default='LinearAttention', help='Classname of one of the models in the models dir')
    p.add_argument('--model_parameters', type=dict, help='dictionary of model parameters')

    p.add_argument('--trainer', type=str, default='binding', help='')
    p.add_argument('--train_sampler', type=str, default=None, help='any of pytorchs samplers or a custom sampler')

    p.add_argument('--eval_on_test', type=bool, default=True, help='runs evaluation on test set if true')
    p.add_argument('--check_se3_invariance', type=bool, default=False, help='check it instead of generating files')
    p.add_argument('--num_confs', type=int, default=1, help='num_confs if using rdkit conformers')
    p.add_argument('--use_rdkit_coords', type=bool, default=None,
                   help='override the rkdit usage behavior of the used model')

    return p.parse_args()


def inference(args, tune_args=None):
    sys.stdout = Logger(logpath=os.path.join(os.path.dirname(args.checkpoint), f'inference.log'), syspart=sys.stdout)
    sys.stderr = Logger(logpath=os.path.join(os.path.dirname(args.checkpoint), f'inference.log'), syspart=sys.stderr)
    seed_all(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == 'cuda' else "cpu")
    transform = transforms.Compose([ToTensor()])
    test_data = PPIDataset(device=device, embeddings_path=args.embeddings,annotations_path=args.test_annotations, remapped_sequences=args.remapped_sequences, transform=transform, **args.dataset_params)
    test_data = move_to_device(test_data, device) 
    print('test size: ', len(test_data))
    model = load_model(args, data_sample=test_data[0], device=device, save_trajectories=args.save_trajectories)
    print('trainable params in model: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
    batch_size = args.batch_size
    batch_size = 1
    # collate_function = globals()[args.collate_function] if args.collate_params == {} else globals()[
    #     args.collate_function](**args.collate_params)

    if len(test_data[0][0].shape) == 2:  # if we have per residue embeddings they have an additional length dim
        collate_function = padded_permuted_collate
    else:  # if we have reduced sequence wise embeddings use the default collate function by passing None
        collate_function = None
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_function)
    checkpoint = torch.load(args.checkpoint, map_location=device)

    model.load_state_dict({k: v for k, v in checkpoint['model_state_dict'].items() if 'cross_coords' not in k})
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    results = []  # prediction and corresponding interaction
    items = []
    false_pred = []
    
    for i, batch in tqdm(enumerate(test_loader)):
        with torch.no_grad():
            *batch, = move_to_device(list(batch), device)
            targets = batch[2]  # the last entry of the batch tuple is always the targets
            metadata = batch[-1]
            item1 = metadata['item1']
            item2 = metadata['item2']
            mask1 = torch.arange(metadata['length1'].max())[None, :] < metadata['length1'][:,None]  # [batchsize, seq_len]
            mask2 = torch.arange(metadata['length2'].max())[None, :] < metadata['length2'][:,None]  # [batchsize, seq_len]
            mask1 = move_to_device(mask1, device)                                                              
            mask2 = move_to_device(mask2, device)

            predictions = model(batch[0], batch[1], mask1, mask2)   # foward the rest of the batch to the model
            loc_pred = torch.max(predictions[..., :2], dim=1)[1]  # get indices of the highest value for loc
            results.append(torch.stack((loc_pred, targets), dim=1).detach().cpu().numpy())

            # false_pred_idx = np.argwhere(loc_pred.detach().cpu().numpy()!=targets.detach().cpu().numpy())
            # for idx in false_pred_idx:
            #     item = item1[idx[0]]+'-'+item2[idx[0]]
            #     false_pred.append(item)
    args.n_draws= 200
    accuracies = []
    mccs = []
    f1s = []
    precision = []
    recall = []
    output_lines = []
    results_concat = np.concatenate(results)

    for i in tqdm(range(args.n_draws)):
        samples = np.random.choice(range(0, len(test_data) - 1), len(test_data))
        sample_res = results_concat[samples]
        accuracies.append(100 * np.equal(sample_res[:, 0], sample_res[:, 1]).sum() / len(sample_res))
        mccs.append(matthews_corrcoef(sample_res[:, 1], sample_res[:, 0]))
        f1s.append(f1_score(sample_res[:, 1], sample_res[:, 0], average='weighted'))
        precision.append(metrics.precision_score(sample_res[:, 1], sample_res[:, 0], zero_division=1))
        recall.append(metrics.recall_score(sample_res[:, 1], sample_res[:, 0], zero_division=1))
        #conf = confusion_matrix(sample_res[:, 1], sample_res[:, 0])
        #class_accuracies.append(np.diag(conf) / conf.sum(1))



    output_lines.append(f"accuracy = {np.mean(accuracies)}")
    output_lines.append(f"mcc = {np.mean(mccs)}")
    output_lines.append(f"f1_score = {np.mean(f1s)}")
    output_lines.append(f"precision = {np.mean(precision)}")
    output_lines.append(f"recall = {np.mean(recall)}")
    output_lines.append(f"accuracy_stderr = {np.std(accuracies)}")
    output_lines.append(f"mcc_stderr = {np.std(mccs)}")
    output_lines.append(f"f1_stderr = {np.std(f1s)}")
    # output_lines.append(f"False Pred = {false_pred}")

    if args.output_directory:
        if not os.path.exists(args.output_directory):
            os.mkdir(args.output_directory)
        if not os.path.exists(f'{args.output_directory}/inference'):
            os.mkdir(f'{args.output_directory}/inference')
        print(f'Writing prediction to {args.output_directory}/inference/inference.txt')
        with open(f'{args.output_directory}/inference/inference.txt', "w") as newfile:
            newfile.writelines("\n".join(output_lines))
    else:
        print(f"We could not found output directory.")

    return false_pred, results_concat

if __name__ == '__main__':
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

    for run_dir in args.run_dirs:
        args.checkpoint = f'runs/{run_dir}/best_checkpoint.pt'
        config_dict['checkpoint'] = f'runs/{run_dir}/best_checkpoint.pt'
        # overwrite args with args from checkpoint except for the args that were contained in the config file
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
        args.model_parameters['noise_initial'] = 0
        false_pred_id, results_test = inference(args)
