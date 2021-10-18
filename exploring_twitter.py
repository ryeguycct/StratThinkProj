# pip install transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup 
# bert model-berttokenizer-adanw and get linear they are part of the bert network training
import torch
#we import specialized deep learning python library
import numpy as np
#we import the numpy library for the algorithms
from sklearn.model_selection import train_test_split
#to split the training set in two, between training and validation
from torch import nn, optim
#to implement the neural network, and optim that is to train the model
from torch.utils.data import Dataset, DataLoader
#to read the dataset and load it
import pandas as pd
#pandas for reading the dataset as it is in csv
from textwrap import wrap
#for display only

#initialization
RANDOM_SEED = 42
#is to initialize the weights randomly but with the same initial values and thus replicate it
MAX_LEN = 280
#the maximum length is set to use 280 twitter characters
BATCH_SIZE = 16
#we will introduce the dataset in 16 packages
DATASET_PATH = '/content/drive/MyDrive/Dataset/training.1600000.processed.Twitter.csv'
NCLASSES = 2
#As I told you in the meeting, I think that as a first test we only identify positive or negative 
#comments, for the following we could increase the difficulty

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
#we use random seed to initialize all the parameters
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#we use "cuda" for the use of the gpu
print(device)
#gpu or cpu test

#load the dataset
from google.colab import drive
#drive access
drive.mount('/content/drive')

df = pd.read_csv(DATASET_PATH, encoding = "ISO-8859-1")
#with pandas we will read the dataset

print(df.head(5))
print(df.shape)
print("\n".join(wrap(df['twitters'][5])))

df.head()

#Token
PRE_TRAINED_MODEL_NAME = "bert-base-cased"
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
#pre training with bert's database

#dataset creation
class IMDBDataset(Dataset):

  def __init__(self,twitters,labels,tokenizer,max_len):
    self.twitters = twitters
    self.labels = labels
    self.tokenizer = tokenizer
    self.max_len = max_len

  def __len__(self):
      return len(self.twitters)
    
  def __getitem__(self, item):
    twitter = str(self.twitters[item])
    label = self.labels[item]
    encoding = tokenizer.encode_plus(
        twitter,
        max_length = self.max_len,
        truncation = True,
        add_special_tokens = True,
        return_token_type_ids = False,
        pad_to_max_length = True,
        return_attention_mask = True,
        return_tensors = 'pt'
        )
    

    return {
          'twitter': twitter,
          'input_ids': encoding['input_ids'].flatten(),
          'attention_mask': encoding['attention_mask'].flatten(),
          'label': torch.tensor(label, dtype=torch.long)
      }


# Data loader:

def data_loader(df, tokenizer, max_len, batch_size):
  dataset = IMDBDataset(
      twitters = df.twitters.to_numpy(),
      labels = df.labels.to_numpy(),
      tokenizer = tokenizer,
      max_len = MAX_LEN
  )

  return DataLoader(dataset, batch_size = BATCH_SIZE, num_workers = 4)

#split dataset
df_train, df_test = train_test_split(df, test_size = 0.5, random_state=RANDOM_SEED)

train_data_loader = data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

# MODEL

class BERTSentimentClassifier(nn.Module):

  def __init__(self, n_classes):
    super(BERTSentimentClassifier, self).__init__()
    self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME, return_dict=False)
    self.drop = nn.Dropout(p=0.3)
    self.linear = nn.Linear(self.bert.config.hidden_size, n_classes)

  def forward(self, input_ids, attention_mask):
    _, cls_output = self.bert(
        input_ids = input_ids,
        attention_mask = attention_mask
    )
    drop_output = self.drop(cls_output)
    output = self.linear(drop_output)
    return output

model = BERTSentimentClassifier(NCLASSES)
model = model.to(device)


# TRAINING
EPOCHS = 5
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(train_data_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps = 0,
    num_training_steps = total_steps
)
loss_fn = nn.CrossEntropyLoss().to(device)

# Training iteration
def train_model(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
  model = model.train()
  losses = []
  correct_predictions = 0
  for batch in data_loader:
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['label'].to(device)
    outputs = model(input_ids = input_ids, attention_mask = attention_mask)
    _, preds = torch.max(outputs, dim=1)
    loss = loss_fn(outputs, labels)
    correct_predictions += torch.sum(preds == labels)
    losses.append(loss.item())
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
  return correct_predictions.double()/n_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):
  model = model.eval()
  losses = []
  correct_predictions = 0
  with torch.no_grad():
    for batch in data_loader:
      input_ids = batch['input_ids'].to(device)
      attention_mask = batch['attention_mask'].to(device)
      labels = batch['label'].to(device)
      outputs = model(input_ids = input_ids, attention_mask = attention_mask)
      _, preds = torch.max(outputs, dim=1)
      loss = loss_fn(outputs, labels)
      correct_predictions += torch.sum(preds == labels)
      losses.append(loss.item())
  return correct_predictions.double()/n_examples, np.mean(losses)

# Training!!!

for epoch in range(EPOCHS):
  print('Epoch {} de {}'.format(epoch+1, EPOCHS))
  print('------------------')
  train_acc, train_loss = train_model(
      model, train_data_loader, loss_fn, optimizer, device, scheduler, len(df_train)
  )
  test_acc, test_loss = eval_model(
      model, test_data_loader, loss_fn, device, len(df_test)
  )
  print('Training: Loss: {}, accuracy: {}'.format(train_loss, train_acc))
  print('Validation: Loss: {}, accuracy: {}'.format(test_loss, test_acc))
  print('')

def classifySentiment(review_text):
  encoding_review = tokenizer.encode_plus(
      review_text,
      max_length = MAX_LEN,
      truncation = True,
      add_special_tokens = True,
      return_token_type_ids = False,
      pad_to_max_length = True,
      return_attention_mask = True,
      return_tensors = 'pt'
      )
  
  input_ids = encoding_review['input_ids'].to(device)
  attention_mask = encoding_review['attention_mask'].to(device)
  output = model(input_ids, attention_mask)
  _, prediction = torch.max(output, dim=1)
  print("\n".join(wrap(review_text)))
  if prediction:
    print('Predicted sentiment: Good')
  else:
    print('Predicted sentiment: Bad')

review_text = "I am very happy to know that I am a human."

classifySentiment(review_text)