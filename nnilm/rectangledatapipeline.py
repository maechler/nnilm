from neuralnilm.data.datapipeline import DataPipeline as NeuralNilmDataPipeline
import numpy as np


class RectangleDataPipeline(NeuralNilmDataPipeline):
    def data_generator(self, fold='train', enable_all_appliances=False, source_id=None, reset_iterator=False, validation=False):
        while True:
            batch = self.get_batch(fold, enable_all_appliances, source_id, reset_iterator, validation)
            X_train = batch.input
            Y_train = np.reshape(batch.target, [self.num_seq_per_batch, 3])

            yield (np.reshape(X_train, [self.num_seq_per_batch, X_train.shape[1], 1]), Y_train.astype(np.float32))
