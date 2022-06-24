import numpy.linalg as LA
import pandas as pd
import numpy as np
import time
import random
import torch
import copy
import os

from torch import nn

from sentence_transformers import SentenceTransformer

from datasets import (load_dataset, load_from_disk,
                     load_metric, ClassLabel)

from transformers import DataCollatorForLanguageModeling, AutoModelForMaskedLM
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                         Trainer, TrainingArguments, DataCollatorWithPadding)


#### EXAMPLE USAGE #####
# import sparsify as sp
# cp = 'bert-base-uncased-finetune-imdb-tweet_eval/checkpoint-1500/'
# dp = 'datasets/imdb/'
# trainer_70 = sp.run_sparsify(70, model_cp=cp, dataPath=dp, num_train_epochs=50, masked=False, token_model = "bert-base-uncased", taskName='un-named')

class SparsifyTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.a
        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        #outputs = model(**inputs)
        
        f_softmax = nn.Softmax(dim=self.axis)
        global model_sp
        # model = model.module if hasattr('module') else model
        
        if self.numSteps_taken[0] % 10 == 0:
            wtsVec = convertTransformer2wts(model)
            _, wts_sp = findVec_sparseHyperplane(wtsVec, self.sparsity)
            model_sp = convertWtsVec_transformer(self.model, wts_sp)
        
        outputs = model(**inputs)
        outputs = f_softmax(outputs.logits)

        outputs_ori = self.model_ori(**inputs)
        outputs_ori = f_softmax(outputs_ori.logits).detach()

        epsAdd = max(1e-10, torch.min(outputs_ori)*1e-3);
        bcloss = -torch.log(torch.sum(torch.sqrt(outputs*outputs_ori+epsAdd), axis=self.axis));

        loss = torch.sum(bcloss);
        
        vecDist = eucDist(model,  model_sp)
        if self.numSteps_taken[0] % 10 == 0:
            print("DIST 2 sparse", vecDist, "SPARSITY", self.sparsity)
        loss = loss + vecDist

        
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

#         if labels is not None:
#             loss = self.label_smoother(outputs, labels)
#         else:
#             # We don't use .loss here since the model may return tuples instead of ModelOutput.
#             loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        self.numSteps_taken[-1] = self.numSteps_taken[-1]+1
        
        print("Time taken for ", self.numSteps_taken[-1], " is = ", time.time() - self.st_time)
        
        #print("number of times running the compute loss fn = ", numSteps_taken[0])
    
        self.outputs = outputs
        return loss

         
    # Tokenize a Dataset.
def tokenize_data(tokenizer, dataset, dataPath):
    tokenDataPath = dataPath +'_token'
    # Check if this dataset has been saved and tokenized before.
    if os.path.exists(tokenDataPath):
        tokenized_datasets = load_from_disk(tokenDataPath)
    # If not tokenize it and save it.
    else:
        features =list(dataset['train'].features.keys())
        if 'label' in features:
            features.remove('label')
        key = features[0]
        tokenized_datasets = dataset.map(lambda examples: tokenizer(examples[key], 
                                                                   max_length=512, 
                                                                   padding="max_length", 
                                                                   truncation=True),
                                                                   batched=True)
        tokenized_datasets.save_to_disk(tokenDataPath)
    return tokenized_datasets

def run_sparsify(sparsity, model_cp, dataPath, num_train_epochs, masked=False, token_model = "bert-base-uncased", taskName='un-named'):
    
    dataset = load_from_disk(dataPath)
    tokenizer = AutoTokenizer.from_pretrained(token_model, use_fast=True)

    tokenized_data = tokenize_data(tokenizer, dataset, dataPath)

    model_name = model_cp.split("/")[-1]

    if masked:
        model = AutoModelForMaskedLM.from_pretrained(model_cp)
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
        axis = 2
    else:
        num_labels = dataset['test'].info.features['label'].num_classes
        model = AutoModelForSequenceClassification.from_pretrained(model_cp, num_labels=num_labels, ignore_mismatched_sizes=True )
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        axis = 1

    training_args = TrainingArguments(
        f"{model_name}-FIP{sparsity}sparse2-{taskName}",
        evaluation_strategy = "epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        push_to_hub=False,
        num_train_epochs= num_train_epochs,
        per_device_train_batch_size=4   
    )
    #print('general', model)
    trainer = SparsifyTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data['train'],
    eval_dataset=tokenized_data['train'].shuffle(seed=42).select(range(1000)),
    data_collator=data_collator,   
    )

#    global model_ori
    trainer.model_ori = copy.deepcopy(model)
    trainer.model_ori.to('cuda:0')
    
    trainer2 = Trainer(
    model=trainer.model_ori, 
    args=training_args,
    train_dataset=tokenized_data['train'],
    eval_dataset=tokenized_data['train'].shuffle(seed=42).select(range(1000)),
    data_collator=data_collator,
    )



 #   global numSteps_taken
    trainer.numSteps_taken = [0]
    trainer.axis = axis
    trainer.sparsity = sparsity
    
    trainer.st_time = time.time()
    
    import math
    eval_results = trainer2.evaluate()
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

    print("CUSTOM TRAINING")
    trainer.train()
    return trainer, eval_results



def convertTransformer2wts(model, layer=-1):
    
    if layer == -1:
        string = 'encoder.'
    else:
        string = 'layer.'+str(layer)

    #layers_dict = {}
    layers_cat = []

    for n, p in model.named_parameters():

        if ((string in n and 'weight' in n) or ('cls' in n and 'weight' in n)) and ('LayerNorm' not in n) and ('decoder' not in n):
            #print(n)

            #layers_dict[n] = p.view(-1,1)
            layers_cat.extend(list(p.view(-1,1).detach().cpu().numpy().flatten()))


    return np.array(layers_cat)

    

# Sparsify p% of vector
def findVec_sparseHyperplane(wtsVec, sparsity):
    
    
    # Find the p% lowest mag weights of the network.
    wts_fin = copy.deepcopy(wtsVec)
    
    numWtsPrune = round(sparsity / 100 * len(wts_fin))
    
    # find min absolute wts and set the mink elements in mask to 0.
    idx = np.argsort(abs(wts_fin))
    
    wts_fin[idx[:numWtsPrune]] = 0.0
    vec_hyperplane = wts_fin - wtsVec
    vec_hyperplane = vec_hyperplane / LA.norm(vec_hyperplane, 2)
    
    return vec_hyperplane, wts_fin
    
    
def convertWtsVec_transformer(model, wtsVec, layer=-1):
    
    model2 = copy.deepcopy(model)
    
    state_dict = model2.state_dict();
    ctr = 0;
    
    if layer == -1:
        string = 'encoder.'
    else:
        string = 'layer.'+str(layer)
    
    for n, p in state_dict.items():
    
        if ((string in n and 'weight' in n) or ('cls' in n and 'weight' in n)) and ('LayerNorm' not in n) and ('decoder' not in n):
            #print(n, p.shape)

            temp = p.view(-1,1);
            temp2 = list(p.size())
            #print(len(temp), n, len(temp2), ctr)

            val = wtsVec[ctr: ctr + len(temp)];
            val = torch.Tensor(val).view(p.shape)

            if len(temp2) == 4:

                for idx in range(val.shape[0]):
                    p[idx,:,:,:] = val[idx,:,:,:]

            elif len(temp2) == 2:

                for idx in range(val.shape[0]):
                    p[idx,:] = val[idx,:]

            else:

                for idx in range(val.shape[0]):
                    p[idx] = val[idx];

            ctr = ctr + len(temp)        

    return model2

def eucDist(model, model_sp, layer=-1):

    vecDist = 0

    if layer == -1:
        string = 'layer.'
    else:
        string = 'layer.'+str(layer)

    for a, b in zip(model.named_parameters(), model_sp.named_parameters()):

        n = a[0]
        if ((string in n and 'weight' in n) or ('cls' in n and 'weight' in n)) and ('LayerNorm' not in n) and ('decoder' not in n):
        #if (string in n and 'weight' in n) and ('LayerNorm' not in n):

            #print(a[0],b[0])
            vecDist = vecDist + torch.norm((a[1]-b[1]),2)

    return vecDist
