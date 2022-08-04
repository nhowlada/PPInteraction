from __future__ import annotations
from typing import Tuple
import sys
import h5py
import torch
from Bio import SeqIO
from torch.utils.data import Dataset
import torch.nn.functional as F
import pandas as pd

from commons.utils import TARGET, AMINO_ACIDS


class PPIDataset(Dataset):
  

    def __init__(self, 
                 device:str = 'cuda:0',
                 embeddings_path:str = '',
                 annotations_path:str = '', 
                 remapped_sequences:str = '', 
                 unknown_solubility: bool = True,
                 key_format:str = 'hash',
                 max_length: int = float('inf'),
                 embedding_mode: str = 'lm',
                 transform=lambda x: x,
                 **kwargs) -> None:
        super().__init__()
        self.transform = transform
        self.embedding_mode = embedding_mode
        if self.embedding_mode == 'lm' or self.embedding_mode == 'profiles':
            self.embeddings_file = h5py.File(embeddings_path, 'r')
        self.annotations_df = pd.read_csv(annotations_path)
        self.annotations_df.set_index(['protein1','protein2'], inplace = True)
        
        self.binding_metadata_list = []
        self.class_weights = torch.zeros(2)
        self.one_hot_enc1 = []
        self.one_hot_enc2 = []

        remapped_dict = SeqIO.to_dict(SeqIO.parse(open(remapped_sequences), 'fasta'), key_function = lambda rec : rec.description.split()[1])

        annotations_dict = self.annotations_df.to_dict(orient="index")

        for (item1,item2), value in annotations_dict.items():
            if item1 in remapped_dict and item2 in remapped_dict:
                record1 = remapped_dict[item1]
                record2 = remapped_dict[item2]
                target = value['label']
                target = TARGET.index(target)  # get TARGET as integer
                id1 = str(record1.id)
                id2 = str(record2.id)
                if len(record1.seq) <= max_length and len(record2.seq) <= max_length:
                    if self.embedding_mode == 'onehot':
                        amino_acid_ids1 = []
                        amino_acid_ids2 = []
                        for char in record1.seq:
                            amino_acid_ids1.append(AMINO_ACIDS[char])
                        for char in record2.seq:
                            amino_acid_ids2.append(AMINO_ACIDS[char])

                        one_hot_enc1 = F.one_hot(torch.tensor(amino_acid_ids1), num_classes=len(AMINO_ACIDS))
                        one_hot_enc2 = F.one_hot(torch.tensor(amino_acid_ids2), num_classes=len(AMINO_ACIDS))
                        self.one_hot_enc1.append(one_hot_enc1)
                        self.one_hot_enc2.append(one_hot_enc2)
                    frequencies1= torch.zeros(25)
                    frequencies2 = torch.zeros(25)
                    for i, aa in enumerate(AMINO_ACIDS):
                        frequencies1[i] = str(record1.seq).count(aa)
                        frequencies2[i] = str(record1.seq).count(aa)
                    frequencies1 /= len(record1.seq)
                    frequencies2 /= len(record1.seq)
                    metadata = {'id1': id1,
                                'id2': id2,
                                'sequence1': str(record1.seq),
                                'sequence2': str(record2.seq),
                                'item1' : item1,
                                'item2' : item2,
                                'length1': len(record1.seq),
                                'length2': len(record2.seq),
                                'frequencies1': frequencies1,
                                'frequencies2': frequencies2
                                }
                    # included sequence to the binding list
                    self.binding_metadata_list.append(
                        {'target': target, 'metadata': metadata})
                self.class_weights[target] += 1
            else:
                print(f"No, Item: '{item1} or {item2}' does not exists in dictionary")
        self.class_weights /= self.class_weights.sum()

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        binding_metadata = self.binding_metadata_list[index]
        if self.embedding_mode == 'lm':
            embedding1 = self.embeddings_file[binding_metadata['metadata']['id1']][:]
            embedding2 = self.embeddings_file[binding_metadata['metadata']['id2']][:]
        elif self.embedding_mode == 'profiles':
            embedding1 = self.embeddings_file[binding_metadata['metadata']['sequence']][:]
            embedding2 = self.embeddings_file[binding_metadata['metadata']['sequence']][:]
        elif self.embedding_mode == 'onehot':
            embedding1 = self.one_hot_enc[index]
            embedding2 = self.one_hot_enc[index]
        else:
            raise Exception('embedding_mode: {} not supported'.format(self.embedding_mode))

        embedding1, embedding2, target = self.transform(
            (embedding1, embedding2, binding_metadata['target']))
        return (embedding1, embedding2, target, binding_metadata['metadata'])

    def __len__(self) -> int:
        return len(self.binding_metadata_list)
