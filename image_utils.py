import torch
import numpy as np


def PILtoTensor(data):
    return torch.from_numpy(np.array(data)).permute(2, 0, 1).float() / 255


def _in_step(config, step):
    return config.start_step <= step < config.end_step


def _classify_blocks(block_list, name):
    return any(block in name for block in block_list)
