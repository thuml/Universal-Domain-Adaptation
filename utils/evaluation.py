import numpy as np

class HScore():
    def __init__(self, unknown_class_index):
        self.unknown_class_index = unknown_class_index

        self.per_class_correct = np.zeros((unknown_class_index+1)).astype(np.float32)
        self.per_class_num = np.zeros((unknown_class_index+1)).astype(np.float32)

    # predictions : (1, )
    # references : (1, )
    def add_batch(self, prediction, reference):

        if prediction == reference:
            self.per_class_correct[reference] += 1
        
        self.per_class_num[reference] += 1
    

    # predictions : (batch, )
    # references : (batch, )
    def add_batch(self, predictions, references):

        batch_size = predictions.shape[0]

        for index in range(batch_size):
            prediction = predictions[index].detach().cpu().numpy()
            reference = references[index].detach().cpu().numpy()

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

        known_accuracy = per_class_accuracy[:valid_count-1].mean() * 100
        unknown_accuracy = per_class_accuracy[valid_count-1] * 100
        h_score = 2 * known_accuracy * unknown_accuracy / (known_accuracy + unknown_accuracy)
            
        total_correct = per_class_correct.sum()
        total_samples = per_class_num.sum()
        accuracy = total_correct / total_samples * 100


        # import pdb
        # pdb.set_trace()

        return {
            'h_score' : h_score,
            'known_accuracy' : known_accuracy,
            'unknown_accuracy' : unknown_accuracy,
            'accuracy' : accuracy
        }