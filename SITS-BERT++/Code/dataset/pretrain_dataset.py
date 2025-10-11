from torch.utils.data import Dataset
import torch
import numpy as np
import random
from dataset.time_warping import time_warping

class PretrainDataset(Dataset):
    def __init__(self, file_path, feature_num, seq_len, apply_time_warping=False):
        """
        :param file_path: pre-training file path
        :param feature_num: number of input features
        :param seq_len: padded sequence length
        :param apply_time_warping: whether to apply time warping augmentation
        """

        self.seq_len = seq_len
        self.dimension = feature_num
        self.apply_time_warping = apply_time_warping

        with open(file_path, 'r') as ifile:
            self.Data = [line.strip() for line in ifile if line.strip()]
            print("Loading data successful")
            self.TS_num = len(self.Data)

    def __len__(self):
        return self.TS_num

    def __getitem__(self, item):
        line = self.Data[item]

        line_data = line.split(',')
        line_data = [float(x) for x in line_data if x]
        line_data = np.array(line_data, dtype=float)

        ts = np.reshape(line_data, (self.dimension + 1, -1)).T
        ts_length = ts.shape[0]

        bert_mask = np.zeros((self.seq_len,), dtype=int)
        bert_mask[:ts_length] = 1

        # BOA reflectances
        ts_origin = np.zeros((self.seq_len, self.dimension))
        ts_origin[:ts_length, :] = ts[:, :-1] / 10000.0

        # day of year
        doy = np.zeros((self.seq_len,), dtype=int)
        doy[:ts_length] = np.squeeze(ts[:, -1])

        # Apply time warping if enabled
        if self.apply_time_warping:
            # print(f"Applying time warping to sample {item}")
            ts_origin = time_warping(ts_origin)

        # randomly add noise to some observations
        ts_masking, mask = self.random_masking(ts_origin, ts_length)

        output = {"bert_input": ts_masking,
                  "bert_target": ts_origin,
                  "bert_mask": bert_mask,
                  "loss_mask": mask,
                  "time": doy,
                  }

        return {key: torch.from_numpy(value) for key, value in output.items()}

    def random_masking(self, ts, ts_length):
        ts_masking = ts.copy()
        mask = np.zeros((self.seq_len,), dtype=int)

        for i in range(ts_length):
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15
                mask[i] = 1

                if prob < 0.5:
                    ts_masking[i, :] += np.random.uniform(low=-0.5, high=0, size=(self.dimension,))
                else:
                    ts_masking[i, :] += np.random.uniform(low=0, high=0.5, size=(self.dimension,))

        return ts_masking, mask