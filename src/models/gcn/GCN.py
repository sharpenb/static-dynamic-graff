import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from torch_geometric.nn import GCNConv
from torch_sparse import SparseTensor


class GCN(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, dropout, lr, weight_decay, seed, evaluator=None):
        super().__init__()
        self.save_hyperparameters()
        torch.cuda.manual_seed(seed)

        self.evaluator = evaluator
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim, cached=True))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_dim))
        for _ in range(n_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim, cached=True))
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim))
        self.convs.append(GCNConv(hidden_dim, output_dim, cached=True))

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.hparams.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)

    def training_step(self, batch, batch_idx):
        edge_index, x, y, indices = batch["edge_index"].squeeze(), batch["x"].squeeze(), batch["y"].long().squeeze(), batch["indices"].squeeze()
        adj_t = SparseTensor(row=edge_index[0], col=edge_index[1]).t()  # TODO check that conversion is OKAY
        out = self.forward(x, adj_t)
        loss = nn.functional.nll_loss(out[indices], y[indices])
        self.log("train_loss", loss)

        return loss

    def predict(self, batch):
        edge_index, x, y, indices = batch["edge_index"].squeeze(), batch["x"].squeeze(), batch["y"].long().squeeze(), batch["indices"].squeeze()
        adj_t = SparseTensor(row=edge_index[0], col=edge_index[1]).t()  # TODO check that conversion is OKAY
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
