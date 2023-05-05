# train auto encoder to reduce dimensionality
from BERT_embedding import BERTEmbedding
from auto_encoder import AutoEncoder
import torch

PATH = "./output/temp/input_data.csv"
INPUT_SHAPE = 768
MODEL_TYPE = 'BERT'


class TrainAE(object):
    def __init__(self, path, vec):
        self.path = path
        self.epochs = 10
        self.vec = vec
        self.model = AutoEncoder()
        self.loss = torch.nn.MSELoss()

    def __len__(self):
        return len(self.vec)

    def auto_encoder_fit(self):
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=1e-1,
                                     weight_decay=1e-8)
        loader = torch.utils.data.DataLoader(dataset=self.vec,
                                             batch_size=32,
                                             shuffle=True)
        losses = []
        outputs = []
        for epoch in range(self.epochs):
            for vec_ite in loader:
                vec_ite = vec_ite.reshape(-1, INPUT_SHAPE)

                # Output of Autoencoder
                reconstructed = self.model(vec_ite)

                # Calculating the loss function
                loss = self.loss(reconstructed, vec_ite)

                # The gradients are set to zero,
                # the gradient is computed and stored.
                # .step() performs parameter update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Storing the losses in a list for plotting
                losses.append(loss)
            outputs.append((self.epochs, vec_ite, reconstructed))
        return outputs


def main():
    bert_obj1 = BERTEmbedding(PATH, MODEL_TYPE)
    vec = bert_obj1.get_sentence_vec()
    training_obj = TrainAE(PATH, vec)
    output = training_obj.auto_encoder_fit()
    reduced = training_obj.model.encoder(torch.from_numpy(vec))
    print(len(output))
    print(len(reduced))


if __name__ == '__main__':
    main()

