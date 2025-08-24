import torch

from model.lior_model_lstm2 import RecommendationModelV2


def load_model(filepath,rna_embedding,amino_embedding,settings):

    model = RecommendationModelV2(rna_embedding.vocab, amino_embedding.vocab, embedding_dim=settings['EMBED_DIM'],
                                  hidden_size=settings['HIDDEN_SIZE'], output_size=1
                                  , rna_n_gram_size=1, amino_n_gram_size=1, lstm_layers=settings['LSTM_LAYER'],
                                  bidirectional=settings['Bidirectional'], fc_units=settings['fc'],
                                  )


    model.load_state_dict(torch.load(filepath))
    return model