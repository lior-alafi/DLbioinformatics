import torch
from torch.utils.data import Dataset



class TextDataset(Dataset):
    def __init__(self, neucleotides_seq, rbp_files, train=False, file3=None):
        self.nec_seqs_len = 0
        self.rpb_seqs_len = 0

        self.nec_filename = neucleotides_seq
        self.rbp_filename = rbp_files

        with open(rbp_files, 'r') as f1:
            data = f1.readlines()
            self.rpb_seqs_len = len(data)
            self.rbp_2_index, self.rbp_2_text, self.rpb_seq_len = self.create_word_to_index(data)
            self.rbp_seqs = self.preprocess(data, self.rbp_2_index, self.rpb_seq_len)
        with open(neucleotides_seq, 'r') as f2:
            data = f2.readlines()
            self.nec_seqs_len = len(data)
            self.nec_2_index, self.nec_2_text, self.nec_seq_len = self.create_word_to_index(data)
            self.nec_seqs = self.preprocess(data, self.nec_2_index, self.nec_seq_len)

        self.length = self.nec_seqs_len * self.rpb_seqs_len

        self.nec_seqs = self.preprocess(data, self.nec_2_index, self.nec_seq_len)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Get item by index.
        """
        rpb_index = idx % self.rpb_seqs_len
        nec_index = idx // self.rpb_seqs_len

        rpb_seq = self.rbp_seqs[rpb_index]
        nec_seq = self.nec_seqs[nec_index]

        return rpb_seq, nec_seq, rpb_index, nec_index

    # def generator_rpb(self):
    #     """
    #     Generate RBP sequences.
    #     """
    #     for i in range(self.rpb_seqs_len):
    #         yield i,self.rbp_2_text[i]

    # def generator_nec(self):
    #     """
    #     Generate nucleotide sequences.
    #     """
    #     for i in range(self.nec_seqs_len):
    #         yield i,self.nec_2_text[i]

    def preprocess(self, data, index_map, max_length):
        """
        Convert a sequence of words into a tensor of indices.
        """
        dataset = []
        for word in data:
            if word == '':
                continue
            # Convert each letter in the word to its corresponding index
            # If the letter is not in the index_map, use 0 (for padding)
            indices = [index_map.get(letter, 0) for letter in word.strip()]
            indices = [0] * (max_length - len(indices)) + indices
            if len(indices) > max_length:
                print(f"Warning: Sequence length {len(indices)} exceeds max length {max_length}. Truncating.")
            dataset.append(indices)

        return torch.tensor(dataset, dtype=torch.long)

    def create_word_to_index(self, data):
        """
        Create a mapping from words to indices.
        """
        to_index = {'<PAD>': 0}
        i = 1
        to_text = {0: '<PAD>'}
        max_length = 0
        for line in data:
            words = line.strip()
            max_length = max(max_length, len(words))
            for letter in words:
                if letter not in to_index:
                    to_index[letter] = i
                    to_text[i] = letter
                    i += 1

        return to_index, to_text, max_length

def create_training_batch(indx, dataset, y, batch_size=64, device='cpu'):
    for i in range(0, len(indx), batch_size):
        AGCU_batch = []
        AA_batch = []
        y_true = []

        curr_batch = indx[i:i+batch_size]
        for j in curr_batch:
            rpb_seq, nec_seq, rpb_index, nec_index = dataset[j]

            AGCU_batch.append(nec_seq)
            AA_batch.append(rpb_seq)

            target = y[nec_index][rpb_index]
            y_true.append(torch.tensor([target], dtype=torch.float32))

        # Stack tensors into batches and send to device
        AGCU_batch = torch.stack(AGCU_batch)
        AA_batch = torch.stack(AA_batch)
        y_true = torch.stack(y_true)

        yield AGCU_batch, AA_batch, y_true