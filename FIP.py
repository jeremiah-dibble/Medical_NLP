import transformers
print(transformers.__version__)

## PREPARE YELP DATASET
import random
import pandas as pd
import numpy as np
import copy
import os

from datasets import (load_dataset, load_from_disk,
                     load_metric, ClassLabel)

from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                         Trainer, TrainingArguments, DataCollatorWithPadding)

from IPython.display import display, HTML
from torch import nn
from torch.utils.data import DataLoader


from transformers import EarlyStoppingCallback



#EXAMPLE USEAGE
# import FIP as fip
# bert_imdb_tweet = fip.FIP(HF_loc='bert-base-uncased', data_locs=['imdb','tweet_eval'], data_args=[None,'hate'], quick=True)
# OR
# import FIP as fip
# bert_imdb_tweet = fip.FIP(HF_loc='bert-base-uncased', data_locs=['imdb','tweet_eval'], data_args=[None,'hate'])
# bert_imdb_tweet.quick_run()




def download_model(instance):
    models_loc = instance.save_loc + "/models/"
    instance.modelPath = models_loc + instance.HF_loc
    if not os.path.exists(instance.modelPath):
        instance.model = SentenceTransformer(instance.HF_loc)
        instance.model.save(instance.modelPath)
    instance.tokenizer = AutoTokenizer.from_pretrained(instance.modelPath, use_fast=True)

def ld(instance, dataset_num):
    instance.dataPath = instance.datsets_loc + instance.data_locs[dataset_num]
    if instance.data_args[dataset_num] != None:
        instance.dataPath +='/'+instance.data_locs[dataset_num]+'_'+instance.data_args[dataset_num]
    if os.path.exists(instance.dataPath):
        instance.dataset = load_from_disk(instance.dataPath)
        print(instance.dataPath)
    else:
        if instance.data_args[dataset_num] == None:
            instance.dataset = load_dataset(instance.data_locs[dataset_num])
        else:
            instance.dataset = load_dataset(instance.data_locs[dataset_num], instance.data_args[dataset_num])
        instance.dataset.save_to_disk(instance.dataPath)
    return instance.dataset

class CustomTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        
        f_softmax = nn.Softmax(dim=1)
        
        inputs_prev = next(iter(prevData_loader))
        inputs_prev = inputs_prev.to("cuda:0")
        
        outputs = model(**inputs_prev)
        outputs = f_softmax(outputs.logits)

        outputs_ori = model_ori(**inputs_prev)
        outputs_ori = f_softmax(outputs_ori.logits).detach()

        epsAdd = max(1e-10, torch.min(outputs_ori)*1e-3);
        bcloss = -torch.log(torch.sum(torch.sqrt(outputs*outputs_ori+epsAdd), axis=1));

        loss = torch.sum(bcloss);
        #print(loss.shape)
        
        op_current = model(**inputs) # Second Dataset (IMBD Data - inputs)
        
        #print(op_current["loss"])
        #print(op_current[0])
        loss = loss + torch.sum(op_current["loss"])
                
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

#         if labels is not None:
#             loss = self.label_smoother(outputs, labels)
#         else:
#             # We don't use .loss here since the model may return tuples instead of ModelOutput.
#             loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        #numSteps_taken[-1] = numSteps_taken[-1]+1
        
        #print("number of times running the compute loss fn = ", numSteps_taken[0])
        return (loss, outputs) if return_outputs else loss

    
class FIP:
    def __init__(self, HF_loc, data_locs, data_args=[None,None], num_train_epochs = 10, save_loc=os.getcwd(), quick=False):

        self.HF_loc = HF_loc
        self.save_loc = save_loc

        self.data_locs = data_locs
        self.data_args = data_args
        self.num_train_epochs = num_train_epochs
  
        
        if quick:
            self.quick_run()
        
    def quick_run(self):
        self.prep_model_datasets()
        
        self.metric = load_metric("accuracy")
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        self.model = self.train_first_model()
        prevData_loader = self.train_second_model()
        
    
    def prep_model_datasets(self):
        download_model(self)
        
        self.datsets_loc = self.save_loc +'/datasets/'
        # Load both datasets and extend the labeling of the second to eliminate overlap
        self.load_datasets()
        self.extend_dataset()

        self.num_labels = (len(np.unique(self.datasets[0]['train']['label'])) + 
                            len(np.unique(self.datasets[1]['train']['label'])))
        print("TOtal # of labels = ", self.num_labels)
        self.hf_model = AutoModelForSequenceClassification.from_pretrained(self.HF_loc, num_labels=self.num_labels)


    # Tokenize a Dataset
    def tokenize_data(self, dataset, dataPath):
        tokenDataPath = dataPath +'_token'
        if os.path.exists(tokenDataPath):
            tokenized_datasets = load_from_disk(tokenDataPath)
        else:
            features =list(dataset['train'].features.keys())
            if 'label' in features:
                features.remove('label')
            key = features[0]
            tokenized_datasets = dataset.map(lambda examples: self.tokenizer(examples[key], 
                                                                                       max_length=512, 
                                                                                    padding="max_length", 
                                                                                       truncation=True),
                                                                                        batched=True)
            tokenized_datasets.save_to_disk(tokenDataPath)
        return tokenized_datasets
        
    #Load and tokenize both datasets    
    def load_datasets(self):
        print("Load datasets")s
        self.dataset_0 = ld(self, 0)
        self.dataPath_0 = self.dataPath
        
        self.token_data_0 = self.tokenize_data(self.dataset_0, self.dataPath_0)
        
        self.dataset_1 = ld(self, 1)
        self.dataPath_1 = self.dataPath
        
        self.token_data_1 = self.tokenize_data(self.dataset_1, self.dataPath_1)
    
        self.datasets = [self.dataset_0, self.dataset_1]
        self.dataPaths = [self.dataPath_0, self.dataPath_1]
        self.token_datasets = [self.token_data_0, self.token_data_1]
    
    #Extned the labeling so there is no overlap in classes between datasets
    def update_label(self, example):
        example['label'] = example['label'] + self.label_len
        return example                     
 
    #Apply te label extention and save the tokenized datasets
    def extend_dataset(self):
        self.label_len = len(np.unique(self.datasets[0]['train']['label']))
        extneded_path = self.dataPaths[1]+'_token_extended'
        if os.path.exists(extneded_path):
            self.token_datasets[1] = load_from_disk(extneded_path)
        else:
            self.token_datasets[1] = self.token_datasets[1].map(self.update_label)
            self.token_datasets[1].save_to_disk(self.dataPaths[1]+'_token_extended')
                                
    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return self.metric.compute(predictions=predictions, references=labels) 
    
    def train_first_model(self):
        print("Train First Model")
        self.model_name = self.HF_loc.split("/")[-1]
        self.data_name0 = self.data_locs[0].split("/")[-1]
        output_path = 'trainer_checkpoint/'
        output_path += f"{self.model_name}-finetune-{self.data_name0}"
        
        training_args = TrainingArguments(
            output_dir=output_path,
            evaluation_strategy = "steps",
            eval_steps = 50, # Evaluation and Save happens every 50 steps
            save_total_limit = 5, # Only last 5 models are saved. Older ones are deleted.
            learning_rate=2e-5,
            weight_decay=0.01,
            push_to_hub=False,
            num_train_epochs=5,#args.num_train_epochs,
            per_device_train_batch_size=8,
            metric_for_best_model = 'accuracy',
            load_best_model_at_end=True
        )

        
        if os.path.exists(output_path):
            check_points = os.listdir(output_path)
            last_cp = output_path+'/'+check_points[-1]
            trainer = AutoModelForSequenceClassification.from_pretrained(last_cp, num_labels=self.num_labels)
        else:
        # TRAIN NETWORK ON FIRST DATASET
            trainer = Trainer(
                        model=self.hf_model,
                        args=training_args,
                        train_dataset=self.token_datasets[0]['train'],
                        eval_dataset=self.token_datasets[0]['test'],
                        data_collator=self.data_collator,
                        tokenizer=self.tokenizer,
                        compute_metrics=self.compute_metrics,
                        callbacks = [EarlyStoppingCallback(early_stopping_patience=2, early_stopping_threshold=2)]
                        )

            trainer.train()
        return trainer   

    def train_second_model(self):
        print("Train Second Model")
        self.data_name1 = self.data_locs[1].split("/")[-1]
        output_path = f"{self.model_name}-finetune-{self.data_name1}"
        training_args = TrainingArguments(
            output_dir=output_path,
            evaluation_strategy = "no",
            save_steps=10,
            learning_rate=2e-5,
            weight_decay=0.01,
            push_to_hub=False,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=8   
        )

        print("CREATING COPY OF MODEL")

        model_ori = copy.deepcopy(self.model) #Model already trained on yelp
        model_ori.to('cuda:0')
        # global numSteps_taken
        # numSteps_taken = [0]


        #Subsample small number of examples from the yelp dataset.
        small_train_first = self.token_datasets[0]['train'].shuffle(seed=42).select(range(2000))
        small_test_first = self.token_datasets[0]['test'].shuffle(seed=42).select(range(1500))

        print("First small dataset", small_test_first)


        #Only used for the Dataloader that will be used from it
        trainer1 = Trainer(
            model=model,
            args=training_args,
            train_dataset=small_train_first,
            eval_dataset=small_test_first,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
            )
        
        self.prevData_loader = trainer1.get_train_dataloader();
        return self.prevData_loader
                            
    def run_custom_trainer(self):
        print('Run Custom Trainer")
        output_path = f"{self.model_name}-finetune-{self.data_name0}-{self.data_name1}"
        self.custom_training_args = TrainingArguments(
            output_dir=output_path,
            evaluation_strategy = "no",
            save_steps=10,
            learning_rate=2e-5,
            weight_decay=0.01,
            push_to_hub=False,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=8   
        )
        
        
        custom_trainer = CustomTrainer(
            model=self.model,
            args=self.custom_training_args,
            train_dataset=self.token_datasets[1]['train'],
            eval_dataset=self.token_datasets[1]['test'],
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
            )

        print("CUSTOM TRAINING")
        custom_trainer.train()
        return custom_trainer
                                
            