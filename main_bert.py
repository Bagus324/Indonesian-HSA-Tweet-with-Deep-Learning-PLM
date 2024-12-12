import sys
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import re
import argparse

from transformers import BertTokenizer

from utils.preprocessor import Preprocessor
from utils.multi_preprocessor import Multi_Preprocessor as pre1
from utils.multi_preprocessor2 import Multi_Preprocessor as pre2_bert
from utils.multi_preprocessor2_bertweet import Multi_Preprocessor as pre2_bertweet
from utils.test_multi_preprocessor2_bertweet import Multi_Preprocessor as pre2_bertweet_test
from utils.multi_preprocessor2_roberta import Multi_Preprocessor as pre2_roberta
from model.multi_label_model_bertcnn import MultiLabelModel as bert_cnn_wps
from model.multi_label_model_wopos import MultiLabelModel as model_wops
from model.multi_label_bertweet import MultiLabelModel as bertweet_wps
from model.multi_label_bertweet_wopos import MultiLabelModel as bertweet_wops
from model.multi_label_bert import MultiLabelModel as bert_wps
from model.multi_label_model_bertweetcnn import MultiLabelModel as bertweet_cnn_wps
from model.multi_label_roberta import MultiLabelModel as roberta_wps
from model.multi_label_model_robertacnn_1layer import MultiLabelModel as roberta_cnn_wps1
from model.multi_label_model_robertacnn_4layer import MultiLabelModel as roberta_cnn_wps4


def clean_str(string):
        string = string.lower()
        string = re.sub(r"[^A-Za-z0-9(),!?\'\-`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\n", "", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        string = string.strip()

        return string

# def predictor(trainer, labels, threshold = 0.5):
#     trained_model = MultiLabelModel.load_from_checkpoint(
#         trainer.checkpoint_callback.best_model_path,
#         labels = labels
#     )
#     trained_model.eval()
#     trained_model.freeze()

#     tokenizer = BertTokenizer.from_pretrained('indolem/indobert-base-uncased')

#     test_comment = "sih kerja delay mulu setan"
#     test_comment = clean_str(test_comment)
#     encoding = tokenizer.encode_plus(
#         test_comment,
#         add_special_tokens = True,
#         max_length = 100,
#         return_token_type_ids = True,
#         padding = "max_length",
#         return_attention_mask = True,
#         return_tensors = 'pt',
#     )

#     test_prediction = trained_model(encoding["input_ids"], 
#                                     encoding["token_type_ids"],
#                                     encoding["attention_mask"])
#     test_prediction = test_prediction.flatten().numpy()
#     print("Prediction for :", test_comment)
#     for label, prediction in zip(labels, test_prediction):
#         if prediction < threshold:
#             continue
#         print(f"{label}: {prediction}")

if __name__ == '__main__':
    dm = pre2_bert()

    labels = [
        'HS', 
        'Abusive', 
        'HS_Individual', 
        'HS_Group', 
        'HS_Religion', 
        'HS_Race', 
        'HS_Physical', 
        'HS_Gender', 
        'HS_Other', 
        'HS_Weak', 
        'HS_Moderate', 
        'HS_Strong'
    ]

    labels_wps = [
        'HS', 
        'Abusive', 
        'HS_Individual', 
        'HS_Group', 
        'HS_Religion', 
        'HS_Race', 
        'HS_Physical', 
        'HS_Gender', 
        'HS_Other', 
        'HS_Weak', 
        'HS_Moderate', 
        'HS_Strong',
        'PS'
    ]
    parser = argparse.ArgumentParser()
    parser.add_argument("-cnn", "--cnn", action="store_true", default=False)
    args = parser.parse_args()
    if args.cnn:
        print("================MODEL CNN ================")
        model = bert_cnn_wps(labels_wps)

        early_stop_callback = EarlyStopping(monitor = 'val_loss', 
                                            min_delta = 0.001,
                                            patience = 3,
                                            mode = "min")
        
        logger = TensorBoardLogger("logs", name="bertcnn")

        trainer = pl.Trainer(accelerator="gpu",
                            devices=1,
                            max_epochs = 15,
                            logger = logger,
                            log_every_n_steps=1,
                            default_root_dir = "./checkpoints/labels",
                            callbacks = [early_stop_callback])

        trainer.fit(model, datamodule = dm)
        trainer.test(model = model, datamodule = dm)
    else:
        print("================MODEL NON CNN ================")
        model = bert_wps(labels_wps)

        early_stop_callback = EarlyStopping(monitor = 'val_loss', 
                                            min_delta = 0.001,
                                            patience = 3,
                                            mode = "min")
        checkpoint_callback = ModelCheckpoint(dirpath=f'./checkpoints/bert', monitor='val_loss', mode='min')
        
        logger = TensorBoardLogger("logs", name="bert")

        trainer = pl.Trainer(accelerator="gpu",
                            devices=1,
                            max_epochs = 15,
                            logger = logger,
                            log_every_n_steps=1,
                            default_root_dir = "./checkpoints/labels",
                            callbacks = [early_stop_callback, checkpoint_callback])

        trainer.fit(model, datamodule = dm)
        trainer.test(model = model, datamodule = dm)
    

    # print("Predictor")
    # predictor(trainer, labels)