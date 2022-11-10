import torch
import pytorc_lightning as pl

class LightingModelWrapper(pl.LightningModule):
    def __init__(self, model, lr, weight_decay, evaluator=None):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.evaluator = evaluator

    def training_step(self, batch, batch_idx):
        xs, y = batch["xs"], batch["y"].long()
        out = self.model(xs)
        loss = nn.functional.nll_loss(out, y)
        self.log("train_loss", loss)

        return loss

    def predict(self, batch):
        xs, y = batch["xs"], batch["y"]
        out = self.model(xs)
        pred = out.max(1)[1]

        return pred, y

    def evaluate(self, y_pred, y_true):
        if self.evaluator:
            acc = self.evaluator.eval({"y_true": y_true.unsqueeze(1), "y_pred": y_pred.unsqueeze(1)})["acc"]
        else:
            acc = y_pred.eq(y_true.squeeze()).sum().item() / y_pred.shape[0]

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
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer