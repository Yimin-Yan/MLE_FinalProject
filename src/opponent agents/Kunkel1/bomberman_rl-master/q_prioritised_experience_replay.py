import numpy as np

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta0=0.4, beta_steps=1e6, epsilon=1e-6):
        self.tree    = SumTree(capacity)
        self.capacity= capacity
        self.alpha   = alpha
        self.beta    = beta0
        self.beta0   = beta0
        self.beta_inc= (1.0 - beta0) / beta_steps
        self.epsilon= epsilon
        self.max_prio = 1.0

    def add(self, experience):
        # always insert with max priority so new samples are likely to be seen
        p = (self.max_prio + self.epsilon) ** self.alpha
        self.tree.add(experience, p)

    def sample(self, batch_size):
        batch, idxs, ps = [], [], []
        total = self.tree.total()
        segment = total / batch_size

        for i in range(batch_size):
            a, b = segment*i, segment*(i+1)
            s = np.random.uniform(a, b)
            idx, p, data = self.tree.get(s)
            batch.append(data)
            idxs.append(idx)
            ps.append(p)

        # compute normalized probabilities
        ps = np.array(ps) / total
        N  = len(self.tree)
        weights = (N * ps) ** (-self.beta)
        weights /= weights.max()

        # anneal beta
        self.beta = min(1.0, self.beta + self.beta_inc)

        return batch, idxs, weights

    def update_priorities(self, idxs, td_errors):
        for idx, err in zip(idxs, td_errors):
            p = (abs(err) + self.epsilon) ** self.alpha
            self.tree.update(idx, p)
            self.max_prio = max(self.max_prio, p)

    def __len__(self):
        return len(self.tree)


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def add(self, data, priority):
        tree_index = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(tree_index, priority)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, tree_index, priority):
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get(self, value):
        parent_index = 0
        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            if left_child_index >= len(self.tree):
                break

            if value <= self.tree[left_child_index]:
                parent_index = left_child_index
            else:
                value -= self.tree[left_child_index]
                parent_index = right_child_index

        data_index = parent_index - self.capacity + 1
        return parent_index, self.tree[parent_index], self.data[data_index]

    def total(self):
        return self.tree[0]

    def __len__(self):
        return self.n_entries
