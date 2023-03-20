import torch
from transformers import DistilBertTokenizer,DistilBertModel
import torch.nn as nn
tokenizer=DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# model=torch.load('model2.pt',map_location='cpu')

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
device=get_default_device()

class DistilClass(nn.Module):
    def __init__(self,num_classes=6):
        super().__init__()
        self.num_classes=num_classes
        self.distbert=dis=DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.pre_classifier=nn.Linear(768,768)
        self.classifier=nn.Linear(768,self.num_classes)
        self.dropout=nn.Dropout(0.2)
        self.relu=nn.Tanh()
  
    def forward(self,input_ids,attention_mask):
        distilbert_out=self.distbert(input_ids,attention_mask=attention_mask)
    # print(distilbert_out.shape)
        hidden_state=distilbert_out[0][:,0,:].to(device)
        pooled_output=self.pre_classifier(hidden_state).to(device)
        pooled_output=self.relu(pooled_output).to(device)
        pooled_output=self.dropout(pooled_output).to(device)
        outputs=self.classifier(pooled_output).to(device)
        return outputs

    def training_step(self,xb):
        inputs,targets=xb
        id=inputs['input_ids'].to(device)
        mask=inputs['attention_mask'].to(device)
        targets=targets.to(device)
        id=id.squeeze(1)
        mask=mask.squeeze(1)
        out=self(id,mask)
    # out=torch.sigmoid(out)
        loss=torch.nn.BCEWithLogitsLoss()(out,targets).to(device)
        return loss
  
    def validation_step(self,xb):
        inputs,targets=xb
        id=inputs['input_ids'].to(device)
        mask=inputs['attention_mask'].to(device)
        targets=targets.to(device)
        id=id.squeeze(1)
        mask=mask.squeeze(1)
        out=self(id,mask)
    # out=torch.sigmoid(out)
        loss=nn.BCEWithLogitsLoss()(out,targets).to(device)
        return {'val_loss':loss}
  
    def validation_epoch_end(self, outputs):  
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        return {'val_loss': epoch_loss.item()}
     
    def epoch_end(self,epoch,result):
        print("Epoch [{}] ,Train_Loss:{:.4f}, Val_Loss:{:.4f}".format(epoch,result['train_loss'],result['val_loss']))

def load_checkpoint(filepath):
      model=DistilClass().to(device)
      model.load_state_dict(torch.load(filepath,map_location='cpu'))
      return model

@torch.no_grad()
def make_prediction(sentence):
  inputs=tokenizer.encode_plus(sentence)
  id=torch.tensor(inputs['input_ids'],dtype=torch.int).to(device)
  mask=torch.tensor(inputs['attention_mask'],dtype=torch.float).to(device)
  id=id.unsqueeze(0)
  mask=mask.unsqueeze(0)
  model=load_checkpoint('./model.pt')
  out=model(id,mask)
  output=torch.sigmoid(out)
  output=(output[0]>0.55).int()
  output=output.detach().cpu().numpy()
  return output
