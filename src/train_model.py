import torch
import torch.nn as nn
from customDataHandler import customDataset
from torch.utils.data import DataLoader
import argparse
from BertTempRel import BertTempRel
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import pandas as pd
import numpy as np 

parser = argparse.ArgumentParser(description='PyTorch TLink Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--train_path', default='EN/samples_3.txt', type=str, help='train data path')
parser.add_argument('--test_path', default='EN/samples_3.txt', type=str, help='test data path')
parser.add_argument('--epochs', default =20, type=int, help="number of epochs")
parser.add_argument('--output_file', default ='train_output', type=str, help="file name to save the accruacy and loss for epochs")
parser.add_argument('--save_model', default='model_x', type=str, help='under which name the trained model should be saved')
parser.add_argument('--batch', default=16, type=int, help='Batch size')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = 'cpu'

root = "../dataset/"

model = BertTempRel(labels=4)
model.to(device)

optimizer = torch.optim.SGD(params=model.parameters(), lr=args.lr)
#optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
loss_fn = nn.CrossEntropyLoss()

train_dataset= customDataset(path=root+args.train_path)
dataloader = DataLoader(train_dataset , batch_size=args.batch , shuffle=True)

def train_loop( ):
    
    total_loss = 0.0
    n_batches = 0
    model.train()
    for batch in dataloader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        loc1 = batch['location1'].to(device)
        loc2 = batch['location2'].to(device)
        y = batch['label'].to(device)
        y_logits = model(input_ids, attention_mask, loc1, loc2)
        optimizer.zero_grad()
        loss = loss_fn(y_logits, y)
        total_loss += loss
        loss.backward()
        optimizer.step()
        n_batches +=1
    return (total_loss/n_batches) 
    


def test(dir):
    test_dataset= customDataset(path=root+dir)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch, shuffle=True)
    y_pred_all = np.empty(0)
    y_true_all = np.empty(0)
    model.eval()
    with torch.inference_mode():
        for batch in test_dataloader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            loc1 = batch['location1'].to(device)
            loc2 = batch['location2'].to(device)
            y = batch['label'].to(device)
            y_logits = model(input_ids, attention_mask, loc1, loc2)
            loss = loss_fn(y_logits, y)
            y_pred = y_logits.argmax(dim=1).detach().cpu().numpy()
            y_true = y.detach().cpu().numpy()
            y_pred_all = np.append(y_pred_all, y_pred)
            y_true_all = np.append(y_true_all, y_true)
        acc = accuracy_score(y_true_all, y_pred_all)
        micro_f1 = f1_score(y_true=y_true_all, y_pred=y_pred_all, average='micro')
        macro_f1 = f1_score(y_true=y_true_all, y_pred=y_pred_all, average='macro')
        CF = confusion_matrix(y_true=y_true_all, y_pred=y_pred_all, labels=[0,1,2,3])
        print(f"Accuracy: {acc} micro-F1: {micro_f1} macro-F1: {macro_f1}\n")
        print(f"----Confusion Matrix-----\n{CF}")

if __name__ == "__main__":
    torch.manual_seed=42
    torch.cuda.manual_seed=42
    train_loss = []
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(args.epochs):
        loss = train_loop()
        print(f"Epoch: {epoch} ########## Train Loss: {loss}\n")
        loss= loss.detach().cpu().numpy()
        train_loss.append(loss)
    output = {'loss': train_loss}
    df = pd.DataFrame(output)
    save_to_path = '../output_files/'+ args.output_file+'.csv' 
    df.to_csv(save_to_path)
    save_to_path = '../saved_models/'+ args.save_model
    torch.save(model.state_dict(), save_to_path)
    test( dir=args.test_path)
    print(f"Model configuration: MLP with SGD Optimizer, model name: {args.save_model}, loss file: {args.output_file}\n")
    print(f"LR: {args.lr}, Epochs: {args.epochs} Batch: {args.batch}\n")
    print("concatenating the cls token as well\n")
    