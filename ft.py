import os
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm.auto import tqdm

import datasets
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    DataCollatorForLanguageModeling,
)

def download_model(instance):
    models_loc = instance.save_loc + "/models/"
    instance.modelPath = models_loc + instance.HF_loc
    if not os.path.exists(instance.modelPath):
        instance.model = SentenceTransformer(instance.HF_loc)
        instance.model.save(instance.modelPath)
    instance.tokenizer = AutoTokenizer.from_pretrained(instance.modelPath)

def ld(instance):
    instance.datsets_loc = instance.save_loc +'/datasets/'
    instance.dataPath = instance.datsets_loc + instance.data_loc
    if instance.data_arg != None:
        instance.dataPath +='/'+instance.data_loc+'_'+instance.data_arg
    if os.path.exists(instance.dataPath):
        instance.dataset = datasets.load_from_disk(instance.dataPath)
        print(instance.dataPath)
    else:
        if instance.data_arg == None:
            instance.dataset = datasets.load_dataset(instance.data_loc)
        else:
            instance.dataset = datasets.load_dataset(instance.data_loc, instance.data_arg)
        instance.dataset.save_to_disk(instance.dataPath)
    return instance.dataset


class FineTune:
    def __init__(self, HF_loc, data_loc, data_arg=None, save_loc=os.getcwd(), 
                 training_args= None, pd_data=None, test_size=.1, short=None):

        self.HF_loc = HF_loc
        self.save_loc = save_loc

        self.data_loc = data_loc
        self.data_arg = data_arg
        
        self.df = pd_data
        self.test_size = test_size
        
        if type(training_args) == type(None):
            self.training_args = TrainingArguments(
            output_dir=self.HF_loc +'_'+self.data_loc,
            #learning_rate=2e-5,
            #per_device_train_batch_size=16,
            #per_device_eval_batch_size=16,
            num_train_epochs=5,
            evaluation_strategy="epoch",
            #weight_decay=0.01,
            )
        else:
            self.training_args = training_args
        
        self.metric = datasets.load_metric("accuracy")
        
        self.short = short
        
    def load_model(self):
        download_model(self)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.modelPath,
                                                                       num_labels=self.num_lab)
        return self.model, self.modelPath
        
    def load_data(self):
        # self.dataset = ld(self)
        return ld(self)
#         self.datsets_loc = self.save_loc +'/datasets/'
#         self.dataPath = self.datsets_loc + self.data_loc
#         if self.data_arg != None:
#             self.dataPath +='/'+self.data_loc+'_'+self.data_arg
#         # self.dataPath = self.datsets_loc + self.data_loc+'_'+self.data_arg
        
#         if os.path.exists(self.dataPath):
#             self.dataset = datasets.load_from_disk(self.dataPath)
#         else:
#             if self.data_arg == None:
#                 self.dataset = datasets.load_dataset(self.data_loc)
#             else:
#                 self.dataset = datasets.load_dataset(self.data_loc, self.data_arg)
#             self.dataset.save_to_disk(self.dataPath)
#         return self.dataset
    
    # Turn a dataframe with a label and text column in that order with any column lables
    # into a data set. The lables will be recast as their index in label_translate.
    
    def group_texts(self, examples):
        # Concatenate all texts
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        # Compute length of concatenated texts
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the last chunk if it's smaller than chunk_size
        total_length = (total_length // self.chunk_size) * self.chunk_size
        # Split by chunks of max_len
        result = {
            k: [t[i : i + self.chunk_size] for i in range(0, total_length, self.chunk_size)]
            for k, t in concatenated_examples.items()
        }
        # Create a new labels column
        result["labels"] = result["input_ids"].copy()
        return result
    
    def mask_tokenize_function(self, examples):
        
        features =list(self.dataset[list(self.dataset.keys())[0]].features.keys())
        if 'label' in features:
            features.remove('label')
        key = features[0]
        result = self.tokenizer(examples[key])
        if self.tokenizer.is_fast:
            result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
        return result
    
    def insert_random_mask(self, batch):
        features = [dict(zip(batch, t)) for t in zip(*batch.values())]
        masked_inputs = self.data_collator(features)
        # Create a new "masked" colSmn for each column in the dataset
        return {"masked_" + k: v.numpy() for k, v in masked_inputs.items()}


    def mask_data(self, chunk_size = 128, mlm_probability=0.15):
        self.chunk_size = 128
        download_model(self)
        self.model = AutoModelForMaskedLM.from_pretrained(self.modelPath)
        self.dataset = ld(self)
        # Use batched=True to  activate fast multithreading!    
        features = list(self.dataset['train'].features.keys())
        self.tokenized_datasets = self.dataset.map(self.mask_tokenize_function, batched=True, remove_columns=features)
        lm_datasets = self.tokenized_datasets.map(self.group_texts, batched=True)

        self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, 
                                                             mlm_probability=mlm_probability)
        dataset = lm_datasets#lm_datasets['unsupervised']
        dataset = dataset.remove_columns(["word_ids"])
        masked_dataset = dataset.map(
            self.insert_random_mask,
            batched=True,
            remove_columns=dataset['test'].column_names,
        )
        masked_dataset = masked_dataset.rename_columns(
            {
                "masked_input_ids": "input_ids",
                "masked_attention_mask": "attention_mask",
                "masked_labels": "labels",
            }
        )
        self.maskedPath = self.datsets_loc +'/masked/'+ self.HF_loc+'/'+self.data_loc
        if self.data_arg != None:
            self.maskedPath +='/'+self.data_loc+'_'+self.data_arg+'_'+
        masked_dataset.save_to_disk(self.maskedPath)
        self.masked_dataset = datasets.load_from_disk(self.maskedPath)
        
        return masked_dataset
    
    def pd_Datasetdict(self):
        # Hugging Face requires the target to be named 'labels'.
        self.df.columns = ['labels', 'text']
        # Set the number of labels to the number of unique labels
        self.num_lab = max(self.dataset['train']['label']) + 1
        # We are translating the codes to their index in a list of the unique codes.
        # This is need to insure the number of labels in the model is equal to the 
        # maximum number that can be given as a label
        self.code_translation = list(set(self.df['labels']))
        translated_codes = []
        for code in self.df['labels']:
            translated_codes.append(self.code_translation.index(code))
        self.df['labels'] =translated_codes 
        #Turn the dataframe into a dataset
        ds = datasets.Dataset.from_pandas(self.df)
        train_dataset, test_dataset= ds.train_test_split(test_size=self.test_size).values()
        #train_dataset, validation_dataset= train_dataset.train_test_split(test_size=0.1).values()
        self.dataset = datasets.DatasetDict({"train":train_dataset,"test":test_dataset,
                                             'label_translate':self.code_translation})
        return self.dataset
    
    def tokenize_data(self):
        features =list(self.dataset['train'].features.keys())
        if 'label' in features:
            features.remove('label')
        key = features[0]
        self.tokenized_datasets = self.dataset.map(lambda examples: self.tokenizer(examples[key], 
                                                                                   max_length=512, 
                                                                                padding="max_length", 
                                                                                   truncation=True),
                                                                                    batched=True)
        return self.tokenized_datasets

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return self.metric.compute(predictions=predictions, references=labels)
    
    def prepare_trainer(self):  
        if type(self.df) == type(None):
            self.load_data()
        else:
            self.pd_Datasetdict()
        print(self.dataset)
        self.num_lab = max(self.dataset['train']['label']) + 1

        self.model, self.modelPath = self.load_model();
        #model = AutoModelForSequenceClassification.from_pretrained(modelPath, num_labels=self.num_lab)
        
        self.tokenized_datasets = self.tokenize_data()

        self.train_dataset = self.tokenized_datasets["train"].shuffle(seed=1)
        self.eval_dataset = self.tokenized_datasets["test"].shuffle(seed=1)
        if self.short == True:
            print('shrink dataset')
            self.train_dataset = self.tokenized_datasets["train"].shuffle(seed=1).select(range(5000))
            self.eval_dataset = self.tokenized_datasets["test"].shuffle(seed=1).select(range(1000))

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
    
        # Run the Trainer
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            compute_metrics=self.compute_metrics,
            data_collator=data_collator,
            tokenizer=self.tokenizer,)
        return self.trainer, self.dataset
    
    def run_trainer(self):
        start = datetime.now()
        self.trainer.train()
        stop = datetime.now()
        seconds = (stop-start)
        rate = seconds/len(self.train_dataset)
        print("This file took: ", seconds)
        print('At a rate of ' + str(rate) +' per line')
        
    def quick_run(self):
        self.prepare_trainer()
        self.run_trainer()
        
        

# def ld(instance):
#     datsets_loc = instance.save_loc +'/datasets/'
#     instance.dataPath = datsets_loc + instance.data_loc
#     if os.path.exists(instance.dataPath):
#         instance.dataset = datasets.load_from_disk(instance.dataPath)
#     else:
#         if instance.data_arg == None:
#             instance.dataset = datasets.load_dataset(instance.data_loc)
#         else:
#             instance.dataset = datasets.load_dataset(instance.data_loc, instance.data_arg)
#         instance.dataset.save_to_disk(instance.dataPath)
#     return instance.dataset


# class GeneralTune:
#     def __init__(self, HF_loc, data_loc, data_arg=None, save_loc=os.getcwd(), 
#                  training_args= None, pd_data=None, test_size=.1, short=None):

#         self.HF_loc = HF_loc
#         self.save_loc = save_loc

#         self.data_loc = data_loc
#         self.data_arg = data_arg
        
#         self.df = pd_data
#         self.test_size = test_size
        
#         if type(training_args) == type(None):
#             self.training_args = TrainingArguments(
#             output_dir=self.HF_loc +'_'+self.data_loc,
#             #learning_rate=2e-5,
#             #per_device_train_batch_size=16,
#             #per_device_eval_batch_size=16,
#             num_train_epochs=5,
#             evaluation_strategy="epoch",
#             #weight_decay=0.01,
#             )
#         else:
#             self.training_args = training_args
        
#         self.metric = datasets.load_metric("accuracy")
        
#         self.short = short
        
#     def load_model(self):
#         models_loc = self.save_loc + "/models/"
#         self.modelPath = models_loc + self.HF_loc
#         self.tokenizer = AutoTokenizer.from_pretrained(self.modelPath)
#         if os.path.exists(self.modelPath):
#             self.model = AutoModelForSequenceClassification.from_pretrained(self.modelPath,
#                                                                             num_labels=self.num_lab)
            
#         else:
#             self.model = AutoModelForSequenceClassification.from_pretrained(self.HF_loc,
#                                                                             num_labels=self.num_lab)
#             model.save(self.modelPath)
#         return self.model, self.modelPath
    
#     def load_data(self):
#         return ld(self)
#     # Turn a dataframe with a label and text column in that order with any column lables
#     # into a data set. The lables will be recast as their index in label_translate.
#     def pd_Datasetdict(self):
#         # Hugging Face requires the target to be named 'labels'.
#         self.df.columns = ['labels', 'text']
#         # Set the number of labels to the number of unique labels
#         self.num_lab = max(self.dataset['train']['label']) + 1
#         # We are translating the codes to their index in a list of the unique codes.
#         # This is need to insure the number of labels in the model is equal to the 
#         # maximum number that can be given as a label
#         self.code_translation = list(set(self.df['labels']))
#         translated_codes = []
#         for code in self.df['labels']:
#             translated_codes.append(self.code_translation.index(code))
#         self.df['labels'] =translated_codes 
#         #Turn the dataframe into a dataset
#         ds = datasets.Dataset.from_pandas(self.df)
#         train_dataset, test_dataset= ds.train_test_split(test_size=self.test_size).values()
#         #train_dataset, validation_dataset= train_dataset.train_test_split(test_size=0.1).values()
#         self.dataset = datasets.DatasetDict({"train":train_dataset,"test":test_dataset,
#                                              'label_translate':self.code_translation})
#         return self.dataset
    
#     def tokenize_data(self):
#         self.tokenized_datasets = self.dataset.map(lambda examples: self.tokenizer(examples['text'], 
#                                                                                    max_length=512, 
#                                                                                    padding="max_length", 
#                                                                                    truncation=True),
#                                                                                     batched=True)
#         return self.tokenized_datasets

#     def compute_metrics(self, eval_pred):
#         logits, labels = eval_pred
#         predictions = np.argmax(logits, axis=-1)
#         return self.metric.compute(predictions=predictions, references=labels)
    
#     def prepare_trainer(self):  
#         if type(self.df) == type(None):
#             self.load_data()
#         else:
#             self.pd_Datasetdict()
#         print(self.dataset)
#         self.num_lab = max(self.dataset['train']['label']) + 1

#         self.model, self.modelPath = self.load_model();
#         #model = AutoModelForSequenceClassification.from_pretrained(modelPath, num_labels=self.num_lab)
        
#         self.tokenized_datasets = self.tokenize_data()

#         self.train_dataset = self.tokenized_datasets["train"].shuffle(seed=1)
#         self.eval_dataset = self.tokenized_datasets["test"].shuffle(seed=1)
#         if self.short == True:
#             print('shrink dataset')
#             self.train_dataset = self.tokenized_datasets["train"].shuffle(seed=1).select(range(5000))
#             self.eval_dataset = self.tokenized_datasets["test"].shuffle(seed=1).select(range(1000))

#         data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
    
#         # Run the Trainer
#         self.trainer = Trainer(
#             model=self.model,
#             args=self.training_args,
#             train_dataset=self.train_dataset,
#             eval_dataset=self.eval_dataset,
#             compute_metrics=self.compute_metrics,
#             data_collator=data_collator,
#             tokenizer=self.tokenizer,)
#         return self.trainer, self.dataset
    
#     def run_trainer(self):
#         start = datetime.now()
#         self.trainer.train()
#         stop = datetime.now()
#         seconds = (stop-start)
#         rate = seconds/len(self.train_dataset)
#         print("This file took: ", seconds)
#         print('At a rate of ' + str(rate) +' per line')
        
#     def quick_run(self):
#         self.prepare_trainer()
#         self.run_trainer()