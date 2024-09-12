import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear


class encoder(nn.Module):
    def __init__(self, n_dim, dims, n_z):
        super(encoder, self).__init__()
        # print(n_dim,dims[0])
        self.enc_1 = Linear(n_dim, dims[0])
        self.enc_2 = Linear(dims[0], dims[1])
        self.enc_3 = Linear(dims[1], dims[2])
        self.z_layer = Linear(dims[2], n_z)
        self.z_b0 = nn.BatchNorm1d(n_z)

    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))
        z = self.z_b0(self.z_layer(enc_h3))
        return z


class decoder(nn.Module):
    def __init__(self, n_dim, dims, n_z):
        super(decoder, self).__init__()
        self.dec_0 = Linear(n_z, n_z)
        self.dec_1 = Linear(n_z, dims[2])
        self.dec_2 = Linear(dims[2], dims[1])
        self.dec_3 = Linear(dims[1], dims[0])
        self.x_bar_layer = Linear(dims[0], n_dim)

    def forward(self, z):
        r = F.relu(self.dec_0(z))
        dec_h1 = F.relu(self.dec_1(r))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)
        return x_bar


class Classifier(nn.Module):
    def __init__(self, nLabel, n_z):
        super(Classifier, self).__init__()
        self.classification1 = Linear(n_z, nLabel)
        self.act = nn.Sigmoid()

    def forward(self, z):
        return self.act(self.classification1(z))


class Autoencoder(nn.Module):

    def __init__(self, n_stacks, n_input, n_z, nLabel):
        super(Autoencoder, self).__init__()

        dims = []
        for n_dim in n_input:

            linshidims = []
            for idim in range(n_stacks - 2):
                linshidim = round(n_dim * 0.8)
                linshidim = int(linshidim)
                linshidims.append(linshidim)
            linshidims.append(1500)
            dims.append(linshidims)

        self.encoder_list = nn.ModuleList([encoder(n_input[i], dims[i], n_z) for i in range(len(n_input))])
        self.decoder_list = nn.ModuleList([decoder(n_input[i], dims[i], n_z) for i in range(len(n_input))])

        self.regression = Linear(n_z, nLabel)
        self.classifier_list = nn.ModuleList([Classifier(nLabel, n_z) for i in range(len(n_input))])

    def forward(self, mul_X):

        z_list = []
        for enc_i, enc in enumerate(self.encoder_list):
            z_i = enc(mul_X[enc_i])
            z_list.append(z_i)

        x_rec_list = []
        for dec_i, dec in enumerate(self.decoder_list):
            x_rec_list.append(dec(z_list[dec_i]))

        y_list = []
        for cla_i, cla in enumerate(self.classifier_list):
            y_list.append(cla(z_list[cla_i]))

        return z_list, x_rec_list, y_list


class Net(nn.Module):

    def __init__(self,
                 n_stacks,
                 n_input,
                 n_z,
                 Nlabel):
        super(Net, self).__init__()

        self.ae = Autoencoder(
            n_stacks=n_stacks,
            n_input=n_input,
            n_z=n_z,
            nLabel=Nlabel)

    def forward(self, mul_X):
        z_list, x_rec_list, y_list = self.ae(mul_X)

        return z_list, x_rec_list, y_list


