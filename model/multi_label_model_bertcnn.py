import sys
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

from sklearn.metrics import classification_report

from transformers import BertModel

from torchmetrics.classification import MultilabelAUROC, MultilabelF1Score, MultilabelAccuracy


class MultiLabelModel(pl.LightningModule):
    def __init__(self,  
                 labels, 
                 lr = 2e-5,
                 embedding_dim = 768,
                 in_channels = 4, 
                 out_channels = 32,
                 num_classes = 13,
                 kernel_size = 10,
                 dropout = 0.3
                 ) -> None:
        super(MultiLabelModel, self).__init__()

        self.lr = lr
        self.labels = labels

        torch.manual_seed(1)
        random.seed(43)

        self.criterion = nn.BCELoss()

        ks = 3

        self.bert_model = BertModel.from_pretrained('indolem/indobert-base-uncased', output_hidden_states = True)
        self.pre_classifier = nn.Linear(768, 768)

        self.conv1 = nn.Conv2d(in_channels, out_channels, (3, embedding_dim), padding=(2, 0), groups=4)
        self.conv2 = nn.Conv2d(in_channels, out_channels, (4, embedding_dim), padding=(3, 0), groups=4)
        self.conv3 = nn.Conv2d(in_channels, out_channels, (5, embedding_dim), padding=(4, 0), groups=4)

        # apply dropout
        self.dropout = nn.Dropout(dropout)
        self.l1 = nn.Linear(ks * out_channels, num_classes)
        self.sigmoid = nn.Sigmoid()

        self.auroc_macro = MultilabelAUROC(num_labels = len(self.labels), average = "macro")
        self.auroc_label = MultilabelAUROC(num_labels = len(self.labels), average = None)
        self.f1_labels = MultilabelF1Score(num_labels=len(self.labels), average='none')
        self.f1_macro = MultilabelF1Score(num_labels=len(self.labels), average='macro')
        self.acc = MultilabelAccuracy(num_labels=len(self.labels), average='micro')
        self.acc_labels = MultilabelAccuracy(num_labels=len(self.labels), average='none')
        # self.auroc = AUROC(num_classes = 12, task = 'multilabel')

    
    def forward(self, input_ids, token_type_ids, attention_mask):
        bert_out = self.bert_model(input_ids = input_ids, 
                                   attention_mask = attention_mask, 
                                   token_type_ids = token_type_ids)
        hidden_state = bert_out[2]
        
        hidden_state = torch.stack(hidden_state, dim = 1)
        hidden_state = hidden_state[:, -4:]

        x = [
            F.relu(self.conv1(hidden_state).squeeze(3)),
            F.relu(self.conv2(hidden_state).squeeze(3)),
            F.relu(self.conv3(hidden_state).squeeze(3))
        ]

        # print(x.shape)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)

        x = self.dropout(x)
        logit = self.l1(x)
        logit = self.sigmoid(logit)
        return logit

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x_input_ids, x_token_type_ids, x_attention_mask, y = train_batch

        out = self(input_ids = x_input_ids,
                   token_type_ids = x_token_type_ids,
                   attention_mask = x_attention_mask)

        # preds = torch.argmax(out, dim = 1)

        loss = self.criterion(out, y.float())
        acc = self.acc(out, y)
        f1_macro = self.f1_macro(out, y)
        auroc_macro = self.auroc_macro(out, y)
        acc_label = self.acc_labels(out, y)
        f1_labels = self.f1_labels(out, y)
        aurocs_label = self.auroc_label(out, y)
 
        metrics = {'train_loss': loss,
                  'train_acc': acc,
                  'train_f1_macro_': f1_macro,
                  'train_auroc_macro': auroc_macro,
                  'train_acc_' + self.labels[0]:acc_label[0],
                  'train_acc_' + self.labels[1]:acc_label[1],
                  'train_acc_' + self.labels[2]:acc_label[2],
                  'train_acc_' + self.labels[3]:acc_label[3],
                  'train_acc_' + self.labels[4]:acc_label[4],
                  'train_acc_' + self.labels[5]:acc_label[5],
                  'train_acc_' + self.labels[6]:acc_label[6],
                  'train_acc_' + self.labels[7]:acc_label[7],
                  'train_acc_' + self.labels[8]:acc_label[8],
                  'train_acc_' + self.labels[9]:acc_label[9],
                  'train_acc_' + self.labels[10]:acc_label[10],
                  'train_acc_' + self.labels[11]:acc_label[11],
                  'train_acc_' + self.labels[12]:acc_label[12],
                  'train_f1_' + self.labels[0]:f1_labels[0],
                  'train_f1_' + self.labels[1]:f1_labels[1],
                  'train_f1_' + self.labels[2]:f1_labels[2],
                  'train_f1_' + self.labels[3]:f1_labels[3],
                  'train_f1_' + self.labels[4]:f1_labels[4],
                  'train_f1_' + self.labels[5]:f1_labels[5],
                  'train_f1_' + self.labels[6]:f1_labels[6],
                  'train_f1_' + self.labels[7]:f1_labels[7],
                  'train_f1_' + self.labels[8]:f1_labels[8],
                  'train_f1_' + self.labels[9]:f1_labels[9],
                  'train_f1_' + self.labels[10]:f1_labels[10],
                  'train_f1_' + self.labels[11]:f1_labels[11],
                  'train_f1_' + self.labels[12]:f1_labels[12],
                  'train_auroc_' + self.labels[0]:aurocs_label[0],
                  'train_auroc_' + self.labels[1]:aurocs_label[1],
                  'train_auroc_' + self.labels[2]:aurocs_label[2],
                  'train_auroc_' + self.labels[3]:aurocs_label[3],
                  'train_auroc_' + self.labels[4]:aurocs_label[4],
                  'train_auroc_' + self.labels[5]:aurocs_label[5],
                  'train_auroc_' + self.labels[6]:aurocs_label[6],
                  'train_auroc_' + self.labels[7]:aurocs_label[7],
                  'train_auroc_' + self.labels[8]:aurocs_label[8],
                  'train_auroc_' + self.labels[9]:aurocs_label[9],
                  'train_auroc_' + self.labels[10]:aurocs_label[10],
                  'train_auroc_' + self.labels[11]:aurocs_label[11],
                  'train_auroc_' + self.labels[12]:aurocs_label[12]
                }
        self.log_dict(metrics, prog_bar=True, on_epoch=True)
     
        return {"loss": loss, "predictions": out, "labels": y}
        
    def validation_step(self, valid_batch, batch_idx):
        x_input_ids, x_token_type_ids, x_attention_mask, y = valid_batch

        out = self(input_ids = x_input_ids,
                   token_type_ids = x_token_type_ids,
                   attention_mask = x_attention_mask)

        # preds = torch.argmax(out, dim = 1)
        
        loss = self.criterion(out, y.float())

        # acc = self.acc(preds, y.float())
        acc = self.acc(out, y)
        f1_macro = self.f1_macro(out, y)
        auroc_macro = self.auroc_macro(out, y)
        acc_label = self.acc_labels(out, y)
        f1_labels = self.f1_labels(out, y)
        aurocs_label = self.auroc_label(out, y)
 
        metrics = {'val_loss': loss,
                  'val_acc': acc,
                  'val_f1_macro_': f1_macro,
                  'val_auroc_macro': auroc_macro,
                  'val_acc_' + self.labels[0]:acc_label[0],
                  'val_acc_' + self.labels[1]:acc_label[1],
                  'val_acc_' + self.labels[2]:acc_label[2],
                  'val_acc_' + self.labels[3]:acc_label[3],
                  'val_acc_' + self.labels[4]:acc_label[4],
                  'val_acc_' + self.labels[5]:acc_label[5],
                  'val_acc_' + self.labels[6]:acc_label[6],
                  'val_acc_' + self.labels[7]:acc_label[7],
                  'val_acc_' + self.labels[8]:acc_label[8],
                  'val_acc_' + self.labels[9]:acc_label[9],
                  'val_acc_' + self.labels[10]:acc_label[10],
                  'val_acc_' + self.labels[11]:acc_label[11],
                  'val_acc_' + self.labels[12]:acc_label[12],
                  'val_f1_' + self.labels[0]:f1_labels[0],
                  'val_f1_' + self.labels[1]:f1_labels[1],
                  'val_f1_' + self.labels[2]:f1_labels[2],
                  'val_f1_' + self.labels[3]:f1_labels[3],
                  'val_f1_' + self.labels[4]:f1_labels[4],
                  'val_f1_' + self.labels[5]:f1_labels[5],
                  'val_f1_' + self.labels[6]:f1_labels[6],
                  'val_f1_' + self.labels[7]:f1_labels[7],
                  'val_f1_' + self.labels[8]:f1_labels[8],
                  'val_f1_' + self.labels[9]:f1_labels[9],
                  'val_f1_' + self.labels[10]:f1_labels[10],
                  'val_f1_' + self.labels[11]:f1_labels[11],
                  'val_f1_' + self.labels[12]:f1_labels[12],
                  'val_auroc_' + self.labels[0]:aurocs_label[0],
                  'val_auroc_' + self.labels[1]:aurocs_label[1],
                  'val_auroc_' + self.labels[2]:aurocs_label[2],
                  'val_auroc_' + self.labels[3]:aurocs_label[3],
                  'val_auroc_' + self.labels[4]:aurocs_label[4],
                  'val_auroc_' + self.labels[5]:aurocs_label[5],
                  'val_auroc_' + self.labels[6]:aurocs_label[6],
                  'val_auroc_' + self.labels[7]:aurocs_label[7],
                  'val_auroc_' + self.labels[8]:aurocs_label[8],
                  'val_auroc_' + self.labels[9]:aurocs_label[9],
                  'val_auroc_' + self.labels[10]:aurocs_label[10],
                  'val_auroc_' + self.labels[11]:aurocs_label[11],
                  'val_auroc_' + self.labels[12]:aurocs_label[12]
                }
        self.log_dict(metrics, prog_bar=True, on_epoch=True)
        
        return loss

    def test_step(self, test_batch, batch_idx):
        x_input_ids, x_token_type_ids, x_attention_mask, y = test_batch
        out = self(input_ids = x_input_ids,
                   token_type_ids = x_token_type_ids,
                   attention_mask = x_attention_mask)
        
        loss = self.criterion(out, y.float())
        acc = self.acc(out, y)
        f1_macro = self.f1_macro(out, y)
        auroc_macro = self.auroc_macro(out, y)
        acc_label = self.acc_labels(out, y)
        f1_labels = self.f1_labels(out, y)
        aurocs_label = self.auroc_label(out, y)
 
        metrics = {'test_loss': loss,
                  'test_acc': acc,
                  'test_f1_macro_': f1_macro,
                  'test_auroc_macro': auroc_macro,
                  'test_acc_' + self.labels[0]:acc_label[0],
                  'test_acc_' + self.labels[1]:acc_label[1],
                  'test_acc_' + self.labels[2]:acc_label[2],
                  'test_acc_' + self.labels[3]:acc_label[3],
                  'test_acc_' + self.labels[4]:acc_label[4],
                  'test_acc_' + self.labels[5]:acc_label[5],
                  'test_acc_' + self.labels[6]:acc_label[6],
                  'test_acc_' + self.labels[7]:acc_label[7],
                  'test_acc_' + self.labels[8]:acc_label[8],
                  'test_acc_' + self.labels[9]:acc_label[9],
                  'test_acc_' + self.labels[10]:acc_label[10],
                  'test_acc_' + self.labels[11]:acc_label[11],
                  'test_acc_' + self.labels[12]:acc_label[12],
                  'test_f1_' + self.labels[0]:f1_labels[0],
                  'test_f1_' + self.labels[1]:f1_labels[1],
                  'test_f1_' + self.labels[2]:f1_labels[2],
                  'test_f1_' + self.labels[3]:f1_labels[3],
                  'test_f1_' + self.labels[4]:f1_labels[4],
                  'test_f1_' + self.labels[5]:f1_labels[5],
                  'test_f1_' + self.labels[6]:f1_labels[6],
                  'test_f1_' + self.labels[7]:f1_labels[7],
                  'test_f1_' + self.labels[8]:f1_labels[8],
                  'test_f1_' + self.labels[9]:f1_labels[9],
                  'test_f1_' + self.labels[10]:f1_labels[10],
                  'test_f1_' + self.labels[11]:f1_labels[11],
                  'test_f1_' + self.labels[12]:f1_labels[12],
                  'test_auroc_' + self.labels[0]:aurocs_label[0],
                  'test_auroc_' + self.labels[1]:aurocs_label[1],
                  'test_auroc_' + self.labels[2]:aurocs_label[2],
                  'test_auroc_' + self.labels[3]:aurocs_label[3],
                  'test_auroc_' + self.labels[4]:aurocs_label[4],
                  'test_auroc_' + self.labels[5]:aurocs_label[5],
                  'test_auroc_' + self.labels[6]:aurocs_label[6],
                  'test_auroc_' + self.labels[7]:aurocs_label[7],
                  'test_auroc_' + self.labels[8]:aurocs_label[8],
                  'test_auroc_' + self.labels[9]:aurocs_label[9],
                  'test_auroc_' + self.labels[10]:aurocs_label[10],
                  'test_auroc_' + self.labels[11]:aurocs_label[11],
                  'test_auroc_' + self.labels[12]:aurocs_label[12]
                }
        self.log_dict(metrics, prog_bar=True, on_epoch=True)


        return out


    # def training_epoch_end(self, outputs):
    #     labels = []
    #     predictions = []
    #     loss = []
    #     for output in outputs:
    #         for out_lbl in output["labels"]:
    #             labels.append(out_lbl)
    #         for out_pred in output["predictions"]:
    #             predictions.append(out_pred)
    #         loss.append(output['loss'])

    #     loss_mean = torch.mean(torch.Tensor(loss))
    #     labels = torch.stack(labels).int()
    #     predictions = torch.stack(predictions)


    #     # print(predictions.shape)
    #     # print(labels.shape)



    #     print(f"Loss Of Epoch {self.current_epoch} : {loss_mean}")


    #     aurocs_label = self.auroc_label(predictions, labels)
    #     auroc_macro = self.auroc_macro(predictions, labels)
    #     accuracy = self.acc(predictions, labels)
    #     self.log("Accuracy", accuracy, prog_bar=True, on_epoch=True)

    #     self.log("Auroc_Macro", auroc_macro, prog_bar=True, on_epoch=True)
    #     f1_labels = self.f1_labels(predictions, labels)
    #     f1_macro = self.f1_macro(predictions, labels)
    #     acc_label = self.acc_labels(predictions, labels)
    #     self.log("F1", f1_macro, prog_bar=True, on_epoch=True)
    #     print("auroc score")
    #     for i, gits in enumerate(aurocs_label) :
    #         print(f"{self.labels[i]} \t : {gits}")
    #         self.logger.experiment.add_scalar(f"{self.labels[i]}_roc_auc/Train", gits, self.current_epoch)
    
    #     print("f1 score")
    #     for i, gits in enumerate(f1_labels) :
    #         print(f"{self.labels[i]} \t : {gits}")
    #         self.logger.experiment.add_scalar(f"{self.labels[i]}_F1/Train", gits, self.current_epoch)

    #     print("Accuracy")
    #     for i, gits in enumerate(acc_label) :
    #         print(f"{self.labels[i]} \t : {gits}")
    #         self.logger.experiment.add_scalar(f"{self.labels[i]}_roc_auc/Train", gits, self.current_epoch)


        # for i, name in enumerate(self.labels):
        #     class_roc_auc = self.auroc_label(predictions[:, i], labels[:, i])

        #     print(f"{name} \t : {class_roc_auc}")

        #     self.logger.experiment.add_scalar(f"{name}_roc_auc/Train", class_roc_auc, self.current_epoch)