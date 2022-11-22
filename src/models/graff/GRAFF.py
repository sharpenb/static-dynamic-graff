import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from torch_geometric.nn.models import MLP
from torch_sparse import SparseTensor
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from src.models.graff.GRAFFConv import GRAFFConv
from src.models.graff.Orthogonal import Orthogonal

class GRAFF(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, output_dim, n_encoder_layers, n_layers, n_decoder_layers, omega_type, Q_type, tau, dropout, lr, weight_decay, seed, W_type, spectral_type=None, temporal_type="static", space_type=None, evaluator=None):
        super().__init__()
        self.save_hyperparameters()
        torch.cuda.manual_seed(seed)

        self.evaluator = evaluator
        self.encoder = MLP(in_channels=input_dim,
                           hidden_channels=hidden_dim,
                           out_channels=hidden_dim,
                           num_layers=n_encoder_layers,
                           dropout=dropout,)
        if self.hparams.spectral_type is not None:
            self.orthogonal_step = Orthogonal(hidden_dim=hidden_dim, orthogonal_map=self.hparams.spectral_type)
        self.graff_steps = torch.nn.ModuleList()
        if self.hparams.temporal_type == 'static':
            graff_step = GRAFFConv(hidden_dim=hidden_dim,
                                   W_type=W_type,
                                   omega_type=omega_type,
                                   Q_type=Q_type,
                                   tau=tau)
            for _ in range(n_layers):
                self.graff_steps.append(graff_step)
        elif self.hparams.temporal_type == 'dynamic':
            for _ in range(n_layers):
                self.graff_steps.append(GRAFFConv(hidden_dim=hidden_dim,
                                                  W_type=W_type,
                                                  omega_type=omega_type,
                                                  Q_type=Q_type,
                                                  tau=tau))
        else:
            raise NotImplementedError
        self.decoder = MLP(in_channels=hidden_dim,
                           hidden_channels=hidden_dim,
                           out_channels=output_dim,
                           num_layers=n_decoder_layers,
                           dropout=dropout,)

    def forward(self, x, adj_t):
        norm_adj = gcn_norm(adj_t.t())

        x = self.encoder(x)
        if self.hparams.spectral_type is not None:
            self.orthogonal_step.forward(x)
        x_0 = x
        for i, graff_step in enumerate(self.graff_steps[:-1]):
            x = graff_step(x, x_0, norm_adj)
            # x = F.relu(x)
            x = F.dropout(x, p=self.hparams.dropout, training=self.training)
        x = self.graff_steps[-1](x, x_0, norm_adj)
        if self.hparams.spectral_type is not None:
            self.orthogonal_step.inverse(x)
        x = self.decoder(x)
        return x.log_softmax(dim=-1)

    def training_step(self, batch, batch_idx):
        edge_index, x, y, indices = batch["edge_index"].squeeze(), batch["x"].squeeze(), batch["y"].long().squeeze(), batch["indices"].squeeze()
        adj_t = SparseTensor(row=edge_index[0], col=edge_index[1]).t()
        out = self.forward(x, adj_t)
        loss = nn.functional.nll_loss(out[indices], y[indices]) # TODO make a function for the loss
        self.log("train_loss", loss)

        return loss

    def predict(self, batch):
        edge_index, x, y, indices = batch["edge_index"].squeeze(), batch["x"].squeeze(), batch["y"].long().squeeze(), batch["indices"].squeeze()
        adj_t = SparseTensor(row=edge_index[0], col=edge_index[1]).t()
        out = self.forward(x, adj_t)
        y_pred = out.max(1)[1]

        return y_pred[indices], y[indices]

    def evaluate(self, y_pred, y_true):
        if self.evaluator:
            acc = self.evaluator.eval({"y_true": y_true.unsqueeze(1), "y_pred": y_pred.unsqueeze(1)})["acc"]
        else:
            acc = y_pred.eq(y_true.squeeze()).sum().item() / y_pred.shape[0] # TODO Maybe use torch metrics

        return acc

    def test_step(self, batch, batch_idx):
        return self.predict(batch)

    def test_epoch_end(self, test_step_outputs):
        y_pred, y_true = zip(*test_step_outputs)
        y_pred = torch.cat(y_pred, dim=0)
        y_true = torch.cat(y_true, dim=0)

        self.log("test_acc", self.evaluate(y_pred, y_true))

    def validation_step(self, batch, batch_idx):
        return self.predict(batch)

    def validation_epoch_end(self, validation_step_outputs):
        y_pred, y_true = zip(*validation_step_outputs)
        y_pred = torch.cat(y_pred, dim=0)
        y_true = torch.cat(y_true, dim=0)

        self.log("val_acc", self.evaluate(y_pred, y_true))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        return optimizer
