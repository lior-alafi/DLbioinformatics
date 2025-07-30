import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class EmbeddingPreprocessor:
    def __init__(self):
        pass
    def preprocess(self,seq):
        pass

class CustomDataset(Dataset):

    def __init__(self, rna_file_path, amino_file_path, scores_file_path='',
                 rna_max_len=41, amino_max_len=912, expected_score_dim=200,merges=None):
        self.expected_score_dim = expected_score_dim  # Store the expected dimension
        self.rna_file_path = rna_file_path
        self.amino_file_path = amino_file_path
        self.scores_file_path = scores_file_path
        self.rna_max_len = rna_max_len
        self.amino_max_len = amino_max_len

        self._load_data()
        self._create_vocabularies()

    def _load_data(self):
        # טעינת קבצי ה-RNA וה-Amino acids
        print("Loading RNA sequences...")
        with open(self.rna_file_path, 'r') as f:
            self.rna_sequences = [line.strip() for line in f]

        print("Loading Amino acid sequences...")
        with open(self.amino_file_path, 'r') as f:
            self.amino_sequences = [line.strip() for line in f]

        # טעינת קובץ ה-Scores
        print("Loading Scores...")
        # נניח שקובץ ה-scores הוא טקסט שמכיל שורות עם ערכי scores.
        # חשוב: אם ה-scores הם מטריצה (לדוגמה, 120678x200), יש לטפל בזה בהתאם.
        # כרגע, מניחים שכל שורה ב-scores היא מחרוזת אחת שבהמשך אולי תומר למספר.
        self.scores = []
        if self.scores_file_path != '':
            with open(self.scores_file_path, 'r') as f:
                for line in f:
                    try:
                        # Split the line by tab ('\t') and convert each part to float
                        # This is the crucial change for scores
                        score_values = [float(val) for val in line.strip().split('\t')]
                        if len(score_values) != self.expected_score_dim:
                            print(
                                f"Warning: Score line has {len(score_values)} values, expected {self.expected_score_dim}. Filling with zeros or truncating.")
                            # Pad or truncate to the expected dimension
                            if len(score_values) < self.expected_score_dim:
                                score_values.extend([0.0] * (self.expected_score_dim - len(score_values)))
                            else:
                                score_values = score_values[:self.expected_score_dim]
                        self.scores.append(score_values)
                    except ValueError as e:
                        print(f"Error parsing score line: '{line.strip()}'. Reason: {e}. Appending zeros.")
                        self.scores.append([0.0] * self.expected_score_dim)  # Append zeros if parsing fails

            # ... (unchanged length verification)
            if len(self.rna_sequences) != len(self.scores):
                raise ValueError(f"Number of RNA sequences ({len(self.rna_sequences)}) "
                                 f"does not match number of scores ({len(self.scores)}).")

        # מספר הדוגמאות הכולל מבוסס על RNA ו-scores (שהם באותו אורך)
        self.num_samples = len(self.rna_sequences)

    def _create_vocabularies(self):
        # RNA (A, C, G, U)
        rna_chars = sorted(list(set(''.join(self.rna_sequences))))
        self.rna_char_to_int = {char: i + 1 for i, char in enumerate(rna_chars)}
        self.rna_char_to_int['<PAD>'] = 0
        self.rna_int_to_char = {i: char for char, i in self.rna_char_to_int.items()}


        amino_chars = sorted(list(set(''.join(self.amino_sequences))))
        self.amino_char_to_int = {char: i + 1 for i, char in enumerate(amino_chars)}
        self.amino_char_to_int['<PAD>'] = 0
        self.amino_int_to_char = {i: char for char, i in self.amino_char_to_int.items()}

        print(f"RNA vocabulary size: {len(self.rna_char_to_int)}")
        print(f"Amino acid vocabulary size: {len(self.amino_char_to_int)}")

    def _sequence_to_int(self, sequence, vocab):
        return [vocab.get(char, 0) for char in sequence]  # השתמש ב-0 עבור תווים לא ידועים

    def __len__(self):
        # מחזיר את מספר הדוגמאות הכולל במערך הנתונים
        return self.num_samples

    def __getitem__(self, idx):
        rna_sequence = self.rna_sequences[idx]
        amino_sequence = self.amino_sequences[idx % len(self.amino_sequences)]
        score_list = []
        if len(self.scores) > 0:
           score_list = self.scores[idx][idx % len(self.amino_sequences)]  # Now score_list is already a list of floats

        rna_numerical = self._sequence_to_int(rna_sequence, self.rna_char_to_int)
        amino_numerical = self._sequence_to_int(amino_sequence, self.amino_char_to_int)

        rna_padded = np.pad(rna_numerical, (self.rna_max_len - len(rna_numerical), 0), 'constant', constant_values=0)
        amino_padded = np.pad(amino_numerical, (self.amino_max_len - len(amino_numerical), 0), 'constant',
                              constant_values=0)

        rna_tensor = torch.tensor(rna_padded, dtype=torch.long)
        amino_tensor = torch.tensor(amino_padded, dtype=torch.long)

        # Convert the list of floats directly to a float tensor
        score_tensor = torch.tensor(score_list, dtype=torch.float)

        return (rna_tensor, amino_tensor), score_tensor