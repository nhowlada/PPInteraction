import numpy as np
import random
from torch.utils.data.sampler import BatchSampler

class BalancedBatchSampler(BatchSampler):
    

    def __init__(self, data_source, batch_size):

        self.target_list = [data_source[i][2].item() for i in range(len(data_source))]
        self.target_to_indices = {target: np.where(np.array(self.target_list) == target)[0]
                                 for target in self.target_list}
        for l in self.target_list:
            np.random.shuffle(self.target_to_indices[l])
        self.used_target_indices_count = {target: 0 for target in self.target_list}
        self.count = 0
        self.batch_size = batch_size
        self.n_classes = len(np.unique(np.array(self.target_list)))
        self.n_samples = int(self.batch_size/self.n_classes)
        self.n_dataset = len(self.target_list)
       
    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            #classes = np.random.choice(self.target_list, self.n_classes, replace=False)
            classes = list(np.unique(np.array(self.target_list)))
            indices = []
            for class_ in classes:
                indices.extend(self.target_to_indices[class_][
                               self.used_target_indices_count[class_]:self.used_target_indices_count[
                                                                         class_] + self.n_samples])
                self.used_target_indices_count[class_] += self.n_samples
                if self.used_target_indices_count[class_] + self.n_samples > len(self.target_to_indices[class_]):
                    np.random.shuffle(self.target_to_indices[class_])
                    self.used_target_indices_count[class_] = 0
            random.shuffle(indices)
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size