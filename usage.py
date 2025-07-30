from torch.utils.data import DataLoader

from data_loader2 import CustomDataset

rna_file = 'data/train_rna_seq.50000.txt'
amino_file = 'data/train_rbps2_seq.50.txt'
scores_file = 'data/train_scores.50000.txt'
batch_size = 32

dataset = CustomDataset(
    rna_file_path=rna_file,
    amino_file_path=amino_file,
    scores_file_path=scores_file,
    rna_max_len=41,
    amino_max_len=912
)

# יצירת DataLoader
# shuffle=True יערבב את הנתונים בכל אפיק
# num_workers > 0 יאפשר טעינת נתונים במקביל (מומלץ לסביבות ייצור)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# מעבר על באצ'ים באמצעות ה-DataLoader
print(f"\nTotal batches: {len(data_loader)}")

for i, ((rna_batch, amino_batch), scores_batch) in enumerate(data_loader):
    print(f"\n--- Batch {i + 1} ---")
    print(f"Shape of rna_batch: {rna_batch.shape}")
    print(f"Shape of amino_batch: {amino_batch.shape}")
    print(f"Shape of scores_batch: {scores_batch.shape}")

    # הצגת דוגמה מהבאצ' הראשון
    if i == 0:
        print("\nExample RNA sequence (first in batch, padded):")
        print(rna_batch[0])
        print("\nExample Amino sequence (first in batch, padded):")
        print(amino_batch[0])
        print("\nExample score (first in batch):")
        print(scores_batch[0])

    if i >= 2:  # נדפיס רק 3 באצ'ים ראשונים לצורך הדוגמה
        break

# גישה למילונים (דרך אובייקט ה-dataset)
print("\nRNA Vocabulary (from dataset object):")
print(dataset.rna_char_to_int)
print("\nAmino Vocabulary (from dataset object):")
print(dataset.amino_char_to_int)