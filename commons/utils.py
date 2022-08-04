
import random
from argparse import Namespace
from collections.abc import MutableMapping
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300

from joblib import Parallel, delayed, cpu_count
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

import dgl
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pad_sequence

from commons.logger import log

def pmap_multi(pickleable_fn, data, n_jobs=None, verbose=1, desc=None, **kwargs):
  if n_jobs is None:
    n_jobs = cpu_count() - 1

  results = Parallel(n_jobs=n_jobs, verbose=verbose, timeout=None)(
    delayed(pickleable_fn)(*d, **kwargs) for i, d in tqdm(enumerate(data),desc=desc)
  )

  return results

def seed_all(seed):
    if not seed:
        seed = 0

    log("[ Using Seed : ", seed, " ]")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    dgl.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def get_random_indices(length, seed=123):
    st0 = np.random.get_state()
    np.random.seed(seed)
    random_indices = np.random.permutation(length)
    np.random.set_state(st0)
    return random_indices

edges_dic = {}
def get_adj_matrix(n_nodes, batch_size, device):
    if n_nodes in edges_dic:
        edges_dic_b = edges_dic[n_nodes]
        if batch_size in edges_dic_b:
            return edges_dic_b[batch_size]
        else:
            # get edges for a single sample
            rows, cols = [], []
            for batch_idx in range(batch_size):
                for i in range(n_nodes):
                    for j in range(n_nodes):
                        rows.append(i + batch_idx*n_nodes)
                        cols.append(j + batch_idx*n_nodes)

    else:
        edges_dic[n_nodes] = {}
        return get_adj_matrix(n_nodes, batch_size, device)

    edges = [torch.LongTensor(rows).to(device), torch.LongTensor(cols).to(device)]
    return edges

def flatten_dict(params: Dict[Any, Any], delimiter: str = '/') -> Dict[str, Any]:
    def _dict_generator(input_dict, prefixes=None):
        prefixes = prefixes[:] if prefixes else []
        if isinstance(input_dict, MutableMapping):
            for key, value in input_dict.items():
                key = str(key)
                if isinstance(value, (MutableMapping, Namespace)):
                    value = vars(value) if isinstance(value, Namespace) else value
                    for d in _dict_generator(value, prefixes + [key]):
                        yield d
                else:
                    yield prefixes + [key, value if value is not None else str(None)]
        else:
            yield prefixes + [input_dict if input_dict is None else str(input_dict)]

    dictionary = {delimiter.join(keys): val for *keys, val in _dict_generator(params)}
    for k in dictionary.keys():
        # convert relevant np scalars to python types first (instead of str)
        if isinstance(dictionary[k], (np.bool_, np.integer, np.floating)):
            dictionary[k] = dictionary[k].item()
        elif type(dictionary[k]) not in [bool, int, float, str, torch.Tensor]:
            dictionary[k] = str(dictionary[k])
    return dictionary




def tensorboard_gradient_magnitude(optimizer: torch.optim.Optimizer, writer: SummaryWriter, step, param_groups=[0]):
    for i, param_group in enumerate(optimizer.param_groups):
        if i in param_groups:
            all_params = []
            for params in param_group['params']:
                if params.grad != None:
                    all_params.append(params.grad.view(-1))
            writer.add_scalar(f'gradient_magnitude_param_group_{i}', torch.cat(all_params).abs().mean(),
                              global_step=step)

def move_to_device(element, device):

    if isinstance(element, list):
        return [move_to_device(x, device) for x in element]
    else:
        return element.to(device) if isinstance(element,(torch.Tensor, dgl.DGLGraph)) else element

def list_detach(element):
    '''
    '''
    if isinstance(element, list):
        return [list_detach(x) for x in element]
    else:
        return element.detach()

def concat_if_list(tensor_or_tensors):
    return torch.cat(tensor_or_tensors) if isinstance(tensor_or_tensors, list) else tensor_or_tensors

def write_strings_to_txt(strings: list, path):
    # every string of the list will be saved in one line
    textfile = open(path, "w")
    for element in strings:
        textfile.write(element + "\n")
    textfile.close()

def read_strings_from_txt(path):
    # every line will be one element of the returned list
    with open(path) as file:
        lines = file.readlines()
        return [line.rstrip() for line in lines]
def padded_permuted_collate(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]]) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """
    """
    embeddings1 = [item[0] for item in batch]
    embeddings2 = [item[1] for item in batch]
    target = torch.tensor([item[2] for item in batch])
    metadata = [item[3] for item in batch]
    metadata = torch.utils.data.dataloader.default_collate(metadata)
    embeddings1 = pad_sequence(embeddings1, batch_first=True)
    embeddings2 = pad_sequence(embeddings2, batch_first=True)
    return embeddings1.permute(0, 2, 1), embeddings2.permute(0, 2, 1), target, metadata

TARGET = ['Negative', 'Positive']
TARGET_ABBREV = ['Neg', 'Pos']

SOLUBILITY = ['M', 'S', 'U']

AMINO_ACIDS = {'A': 0,
               'R': 1,
               'N': 2,
               'D': 3,
               'C': 4,
               'Q': 5,
               'E': 6,
               'G': 7,
               'H': 8,
               'I': 9,
               'L': 10,
               'K': 11,
               'M': 12,
               'F': 13,
               'P': 14,
               'S': 15,
               'T': 16,
               'W': 17,
               'Y': 18,
               'V': 19,
               'U': 20,
               'X': 21,
               'B': 22,
               'J': 23,
               'Z': 24}

def plot_class_accuracies(results, path, args=None):
    confusion = confusion_matrix(results[:, 1], results[:, 0], normalize='true')  # confusion matrix for train
    labels = TARGET

    class_accuracies = np.diag(confusion) / confusion.sum(1)
    class_accuracies_df = pd.DataFrame({'Interaction': labels,"Accuracy": class_accuracies})

    sn.set_style('darkgrid')
    barplot = sn.barplot(x="Accuracy", y="Interaction", data=class_accuracies_df, ci=None)
    barplot.set(xlabel='Accuracy', ylabel='')
    barplot.axvline(1)
    # plt.errorbar(x=df['Accuracy'], y=labels, xerr=df['std'],
    #              fmt='none', c='black', capsize=3)
    plt.tight_layout()
    plt.savefig(path)
    plt.clf()

def plot_confusion_matrix(results, path):
    confusion = confusion_matrix(results[:, 1], results[:, 0], normalize='true')  # normalize='true' for relative freq
    confusion = np.array(confusion, dtype=float)
    confusion[confusion < 0.01] = np.nan
    # confusion[confusion == 0.] = np.nan
    confusion_df = pd.DataFrame(confusion, TARGET_ABBREV, TARGET_ABBREV)
    sn.set_style("whitegrid")

    # fmt='.2f' for relative freq
    # fmt='g' for absolute freq
    sn.heatmap(confusion_df, annot=True, cmap='gray_r',
            fmt='.2f', rasterized=False, cbar=False)
    plt.savefig(path)
    plt.clf()

def tensorboard_class_accuracies(train_results: np.ndarray, val_results: np.ndarray, writer: SummaryWriter, args,
                                 step: int):
    """
    """
    train_confusion = confusion_matrix(
        train_results[:, 1], train_results[:, 0])  # confusion matrix for train
    # confusion matrix for validation
    val_confusion = confusion_matrix(val_results[:, 1], val_results[:, 0])
    labels = TARGET

    train_class_accuracies = np.diag(train_confusion) / train_confusion.sum(1)
    val_class_accuracies = np.diag(val_confusion) / val_confusion.sum(1)
    train_class_accuracies = pd.DataFrame({'Localization': labels,
                                           "Accuracy": train_class_accuracies})
    val_class_accuracies = pd.DataFrame({'Localization': labels,
                                         "Accuracy": val_class_accuracies})
    sn.set_style('darkgrid')
    fig, ax = plt.subplots(1, 2, figsize=(15, 6.5))
    ax[0].set_title('Training')
    ax[1].set_title('Validation')
    barplot1 = sn.barplot(x="Accuracy", y="Localization",
                          ax=ax[0], data=train_class_accuracies, ci=None)
    barplot1.set(xlabel='Accuracy', ylabel='')
    barplot1.axvline(1)
    barplot2 = sn.barplot(x="Accuracy", y="Localization",
                          ax=ax[1], data=val_class_accuracies, ci=None)
    barplot2.set(xlabel='Accuracy', ylabel='')
    barplot2.axvline(1)
    plt.tight_layout()
    writer.add_figure('Class accuracies ', fig, global_step=step)


def tensorboard_confusion_matrix(train_results: np.ndarray, val_results: np.ndarray, writer: SummaryWriter, args,
                                 step: int):
    """
    """
    train_confusion = confusion_matrix(
        train_results[:, 1], train_results[:, 0])  # confusion matrix for train
    # confusion matrix for validation
    val_confusion = confusion_matrix(val_results[:, 1], val_results[:, 0])
    train_cm = pd.DataFrame(train_confusion, TARGET_ABBREV, TARGET_ABBREV)
    val_cm = pd.DataFrame(val_confusion, TARGET_ABBREV, TARGET_ABBREV)

    fig, ax = plt.subplots(1, 2, figsize=(15, 6.5))
    ax[0].set_title('Training')
    ax[1].set_title('Validation')
    sn.heatmap(train_cm, ax=ax[0], annot=True,
               cmap='Blues', fmt='g', rasterized=False)
    sn.heatmap(val_cm, ax=ax[1], annot=True,
               cmap='YlOrBr', fmt='g', rasterized=False)
    writer.add_figure('Confusion Matrix ', fig, global_step=step)
