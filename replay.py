from collections import defaultdict

import numpy as np
import torch


class Replay:
    def __init__(self, size, batch_size, device):
        self.MEMORY_CAPACITY = size
        self.memory_counter = 0
        self.BATCH_SIZE = batch_size
        self.cuda_info = device is not None

    def _sample(self):
        sample_index = np.random.choice(self.MEMORY_CAPACITY, self.BATCH_SIZE)
        return sample_index

    def sample(self):
        raise NotImplementedError()

    def store_transition(self, resource):
        raise NotImplementedError()


class RandomClusterReplay(Replay):
    def __init__(self, size, batch_size, state_shape, device):
        super().__init__(size, batch_size, device)
        self.mem1 = torch.zeros(self.MEMORY_CAPACITY, state_shape)
        self.mem2 = torch.zeros(self.MEMORY_CAPACITY, state_shape)
        self.action = torch.zeros(self.MEMORY_CAPACITY, 1)
        self.reward = torch.zeros(self.MEMORY_CAPACITY, 1)
        if self.cuda_info:
            self.mem1 = self.mem1.cuda()
            self.mem2 = self.mem2.cuda()
            self.action = self.action.cuda()
            self.reward = self.reward.cuda()
        self.alist = [None] * self.MEMORY_CAPACITY

    def store_transition(self, mems):
        s1, action, r, s2, alist = mems
        index = self.memory_counter % self.MEMORY_CAPACITY
        self.mem1[index] = s1
        self.action[index] = action
        self.mem2[index] = s2
        self.reward[index] = r
        self.alist[index] = alist
        self.memory_counter += 1

    def sample(self):
        sample_index = self._sample()
        alist = [self.alist[i] for i in sample_index]
        return self.mem1[sample_index], self.action[sample_index], self.reward[sample_index], self.mem2[sample_index], alist


class RandomOperationReplay(Replay):
    def __init__(self, size, batch_size, state_dim, device):
        super().__init__(size, batch_size, device)
        self.STATE_DIM = state_dim
        self.mem1 = torch.zeros(self.MEMORY_CAPACITY, self.STATE_DIM)
        self.mem2 = torch.zeros(self.MEMORY_CAPACITY, self.STATE_DIM)
        self.op = torch.zeros(self.MEMORY_CAPACITY, 1)
        self.reward = torch.zeros(self.MEMORY_CAPACITY, 1)
        if self.cuda_info:
            self.mem1 = self.mem1.cuda()
            self.mem2 = self.mem2.cuda()
            self.op = self.op.cuda()
            self.reward = self.reward.cuda()

    def store_transition(self, mems):
        s1, op, r, s2 = mems
        index = self.memory_counter % self.MEMORY_CAPACITY
        self.mem1[index] = s1
        self.mem2[index] = s2
        self.reward[index] = r
        self.op[index] = op
        self.memory_counter += 1

    def sample(self):
        sample_index = self._sample()
        return self.mem1[sample_index], self.op[sample_index], self.reward[sample_index], self.mem2[sample_index]


class PERClusterReplay(RandomClusterReplay):
    def __init__(self, size, state_dim, action_dim, batch_size):
        super().__init__(size, state_dim, action_dim, batch_size)

    def _sample(self):
        raise NotImplementedError()
