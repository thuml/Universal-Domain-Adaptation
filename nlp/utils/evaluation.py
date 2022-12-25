import numpy as np
import torch

import torch.nn.functional as F

class HScore():
    def __init__(self, unknown_class_index):
        self.unknown_class_index = unknown_class_index

        self.per_class_correct = np.zeros((unknown_class_index+1)).astype(np.float32)
        self.per_class_num = np.zeros((unknown_class_index+1)).astype(np.float32)
    
    # predictions : (batch, )
    # references : (batch, )
    def add_batch(self, predictions, references):

        batch_size = predictions.shape[0]

        for index in range(batch_size):
            # for torch.tensor
            if 'datach' in dir(predictions):
                prediction = predictions[index].detach().cpu().numpy()
                reference = references[index].detach().cpu().numpy()
            # for numpy.ndarray
            else:
                prediction = predictions[index]
                reference = references[index]

            if prediction == reference:
                self.per_class_correct[reference] += 1
            
            self.per_class_num[reference] += 1


    def compute(self):
        # number of valid classes
        valid_index = ~(self.per_class_num == 0)
        valid_count = np.count_nonzero(valid_index)
        per_class_correct = self.per_class_correct[valid_index]
        per_class_num = self.per_class_num[valid_index]

        per_class_accuracy = per_class_correct / per_class_num

        mean_accuracy = per_class_accuracy.mean() * 100
        known_accuracy = per_class_accuracy[:valid_count-1].mean() * 100
        unknown_accuracy = per_class_accuracy[valid_count-1] * 100
        h_score = 2 * known_accuracy * unknown_accuracy / (known_accuracy + unknown_accuracy)
            
        total_correct = per_class_correct.sum()
        total_samples = per_class_num.sum()
        total_accuracy = total_correct / total_samples * 100

        return {
            'h_score' : h_score,
            'known_accuracy' : known_accuracy,
            'unknown_accuracy' : unknown_accuracy,
            'mean_accuracy' : mean_accuracy,
            'total_accuracy' : total_accuracy
        }

class Accuracy:
    def __init__(self):
        self.predictions = []
        self.references = []
        self.num_samples = 0

    def add_batch(self, predictions, references):
        prediction_count = predictions.shape[0]
        reference_count = references.shape[0]
        assert prediction_count == reference_count, f'{prediction_count} != {reference_count}'
        self.num_samples += prediction_count

        self.predictions.append(predictions)
        self.references.append(references)

        assert len(self.predictions) == len(self.references), f'{len(self.predictions)} != {len(self.references)}'
    
    def compute(self):
        # shape : (num_samples, )
        predictions = torch.concat(self.predictions)
        # shape : (num_samples, )
        references = torch.concat(self.references)

        correct = predictions == references

        num_total = len(correct)
        num_correct = len(correct[correct == True])
        accuracy = num_correct / num_total * 100


        assert num_total == self.num_samples, f'{num_total} != {self.num_samples}'

        return {
            'accuracy' : accuracy,
            'num_samples' : self.num_samples
        }


# for entropy minimization
class entropy(torch.nn.Module):
    def __init__(self):
        super(entropy, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum()
        return b