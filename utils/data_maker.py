import os


def create_n_seqs(filename,new_file,n,start=0,score_range=[]):
    """
    Create a file with n RNA sequences.
    
    :param filename: The name of the file to create.
    :param n: The number of RNA sequences to generate.
    """
    i = 0
    with open(filename, 'r') as f:
        with open(f'{new_file}.{n}.txt', 'w') as out_f:
            for line in f.readlines():
                seq = line.strip()
                if seq and start <= i < start + n:
                        if len(score_range)==0:
                            out_f.write(seq + '\n')
                        else:
                            s = seq.split('\t')
                            out_f.write('\t'.join(s[score_range[0]:score_range[0]+score_range[1]]) + '\n')
                i += 1


if __name__ == "__main__":
    print(os.curdir)
    # create_n_seqs('training_seqs.txt','train_rna_seq', 50000)
    # create_n_seqs('training_seqs.txt','validation_rna_seq', 12500,50000)
    # create_n_seqs('training_RBPs2.txt','train_rbps2_seq', 50)
    # create_n_seqs('training_RBPs2.txt','validation_rbps2_seq', 13,50)
    # create_n_seqs('training_data2.txt','train_scores', 50000,score_range=[0,50])
    # create_n_seqs('training_data2.txt','validation_scores', 12500,50000,score_range=[50,13])
    create_n_seqs('../data/training_seqs.txt', 'train_rna_seq', 102578)
    create_n_seqs('../data/training_seqs.txt', 'validation_rna_seq', 18100, 102578)
    create_n_seqs('../data/training_RBPs2.txt', 'train_rbps2_seq', 170)
    create_n_seqs('../data/training_RBPs2.txt', 'validation_rbps2_seq', 30, 170)
    create_n_seqs('../data/training_data2.txt', 'train_scores', 102578, score_range=[0, 170])
    create_n_seqs('../data/training_data2.txt', 'validation_scores', 18100, 102578, score_range=[170, 30])


    # create_n_seqs('test_data2.txt','test_scores', 102578,score_range=[0,20])