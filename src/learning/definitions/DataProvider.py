class DataProvider:
    def __init__(self, data, labels, batch_size=1, shuffle=True):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(data)
        self.num_batches = self.num_samples // self.batch_size
        self.reset()

    def reset(self):
        self.index = 0
        if self.shuffle:
            self.shuffle_data()

    def shuffle_data(self):
        perm = np.random.permutation(self.num_samples)
        self.data = self.data[perm]
        self.labels = self.labels[perm]

    def next_batch(self):
        if self.index + self.batch_size > self.num_samples:
            self.reset()
        batch_data = self.data[self.index:self.index + self.batch_size]
        batch_labels = self.labels[self.index:self.index + self.batch_size]
        self.index += self.batch_size
        return batch_data, batch_labels

    def get_all_data(self):
        return self.data, self.labels