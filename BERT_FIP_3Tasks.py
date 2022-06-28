import transformers
print(transformers.__version__)

## PREPARE YELP DATASET
import random
import pandas as pd
import numpy as np
import copy
import os
from sentence_transformers import SentenceTransformer

from datasets import (load_dataset, load_from_disk, DatasetDict,
                     load_metric, ClassLabel, concatenate_datasets)

from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                         Trainer, TrainingArguments, DataCollatorWithPadding)

from IPython.display import display, HTML
import torch
from torch import nn, cuda, device
from torch.utils.data import DataLoader


from transformers import EarlyStoppingCallback

import gc



#EXAMPLE USEAGE
# import FIP as fip
# bert_imdb_tweet = fip.FIP(HF_loc='bert-base-uncased', 
#         data_locs=['imdb','tweet_eval'], data_args=[None,'hate'], quick=True)

# OR
# import FIP as fip 
# bert_imdb_tweet = fip.FIP(HF_loc='bert-base-uncased', 
#                     data_locs=['imdb','tweet_eval'], data_args=[None,'hate'])
# bert_imdb_tweet.quick_run()
#
# Too add a third Task
# import FIP_Bert as fip
# bert_imdb_tweet = fip.FIP(HF_loc='bert-base-uncased', data_locs=['imdb','tweet_eval'], data_args=[None,'hate'], quick=False, short=True)
# bert_imdb_tweet.additionalt_task(data_loc='emotion', data_arg='hate', 
#               two_ct_checkpoint='bert-base-uncased-finetune-imdb-tweet_eval/checkpoint-1000/',load=False)

# Downlad a model and define the tokenizer.
def download_model(instance):
    # 
    models_loc = instance.save_loc + "/models/"
    # Check if the model exits and only download if not.f
    instance.modelPath = models_loc + instance.HF_loc
    if not os.path.exists(instance.modelPath):
        instance.model = SentenceTransformer(instance.HF_loc)
        instance.model.save(instance.modelPath)
    instance.tokenizer = AutoTokenizer.from_pretrained(instance.modelPath,
                                                      use_fast=True)
# Load one of the two datasets.
def ld(instance, dataset_num):
    instance.dataPath = instance.datsets_loc + instance.data_locs[dataset_num]
    # If there is a data arg add it save location name.
    if instance.data_args[dataset_num] != None:
        instance.dataPath +='/'+instance.data_locs[dataset_num]+'_'\
                                    +instance.data_args[dataset_num]
    # Check if the dataset has been downloaded before and load it localy if so.
    if os.path.exists(instance.dataPath):
        instance.dataset = load_from_disk(instance.dataPath)
        print(instance.dataPath)
    else:
        if instance.data_args[dataset_num] == None:
            instance.dataset = load_dataset(instance.data_locs[dataset_num])
        else:
            instance.dataset = load_dataset(instance.data_locs[dataset_num],
                                           instance.data_args[dataset_num])
        instance.dataset.save_to_disk(instance.dataPath)
    return instance.dataset

# This is the  FIP CustomTrainer
class CustomTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return
                                            the loss in the first element.
        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        
        f_softmax = nn.Softmax(dim=1)
        
        inputs_prev = next(iter(prevData_loader))
        #inputs_prev = inputs_prev.to("cuda:0")
        for key in inputs_prev:
            inputs_prev[key] = inputs_prev[key].to("cuda:0")

        
        
        outputs = model(**inputs_prev)
        outputs = f_softmax(outputs.logits)

        outputs_ori = model_ori(**inputs_prev)
        outputs_ori = f_softmax(outputs_ori.logits).detach()

        epsAdd = max(1e-10, torch.min(outputs_ori)*1e-3);
        bcloss = -torch.log(torch.sum(torch.sqrt(outputs*outputs_ori+epsAdd),
                                      axis=1));

        loss = torch.sum(bcloss);
        #print(loss.shape)
        
        op_current = model(**inputs) # Second Dataset (IMBD Data - inputs)
        
        #print(op_current["loss"])
        #print(op_current[0])
        loss = loss + torch.sum(op_current["loss"])
                
        # Save past state if it exists.
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        return (loss, outputs) if return_outputs else loss

    
class FIP:
    '''
        FIP requires a model and two datasets, some datasets require optional
     arguments.
        HF_loc should be a classification model's local disk location or 
     huggingface locations. (HF example: 'emilyalsentzer/Bio_ClinicalBERT')
        data_locs must be a list of two dataset local disk location or 
     huggingface locations.
        data_args must be a list containing 'None' or an option of their 
     respective data_locs datasubsets subsets. 
     (example: dataset_loc[0]=glue, data_args[0] = 'cola')
        num_train_epochs is the number of training epochs. Default is 50.
        save_loc is the base directory that models and datasets will be 
        saved to. 
        quick=True automatically runs the quick_run() method after 
        initialization.
        short=True subsamples the datasets to a 5k training and 2k testset.
        Generally only useful for debugging.
    '''
    def __init__(self, HF_loc, data_locs, data_args=[None,None], 
                 num_train_epochs = 50, save_loc = os.getcwd(),
                 quick = False, short = False):
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Assining inputs to the class instance.
        self.HF_loc = HF_loc
        self.save_loc = save_loc

        self.datsets_loc = self.save_loc +'/datasets/'
        self.data_locs = data_locs
        self.data_args = data_args
        
        self.num_train_epochs = num_train_epochs
        self.short = short
        self.two_ct = None
        
        self.model_name = self.HF_loc.split("/")[-1]
        # Grab the datat name.
        self.data_name0 = self.data_locs[0].split("/")[-1]
        self.data_name1 = self.data_locs[1].split("/")[-1]
        if quick:
            self.two_ct =  self.two_task()


    '''

    This method will take a FIP model for two tasks and add a third task.
    This is acomplished by combining the first two tasks into one dataset 
    and using a two taks FIP model as the starting model. Then simply 
    continue like a two task FIP.
    For this to work the two taks FIP must already have been run or
    a checkpoint from the two task custom trainer can be passed.

    Arguments
    data_loc: A huggingface data location or a location on disk within
              the 'save_loc/datasets directory.
              (E.g. 'Glue' or 'Emotion')
    data_arg: The required datasubset selection from data_loc's subsets.
              If data_loc does not have subsets data_arg should be left
              as None
              (E.g. 'Cola' or None)
    two_ct_checkpoint: If the additional task is not being added after
                       two trask training a twotask model checkpoint 
                       must be passed.
                       (E.g. 'bert-base-uncased-finetune-imdb-tweet_eval/checkpoint-1000/')
    load: If True the method will atempt to load the combined 
          dataset if it has been created in the past. If False
          the method will recreate teh combined dataset each run
    '''
    def additionalt_task(self, data_loc, data_arg=None, two_ct_checkpoint=None, load=True):\
        # Load the two task datasets
        dataset_0 = ld(self, 0)
        dataset_1 = ld(self, 1)
        
        

        # Create a save location for the combined datasets.
        dataPath = self.datsets_loc 
        pathExtention =  'combined_datasets/'

        pathExtention += self.data_name0
        if self.data_args[0] != None:
            pathExtention += '_'+self.data_args[0]

        pathExtention += '_' + self.data_name1
        if self.data_args[1] != None:
            pathExtention += '_' + self.data_args[1]

        self.combined_dataPath = dataPath + pathExtention
        # If we have a dataset saved at this location and load is true
        # we will load from disk rather than recreating the dataset.
        if os.path.exists(self.combined_dataPath) and load:
            self.two_task_dataset = load_from_disk(self.combined_dataPath)
        else:
            # Extend the class lables so there is no overlap.
            # Make both datasetsa the same length
            # Combine the two datasets and save the result. 
            self.combine_dataset(dataset_0, dataset_1)
            self.two_task_dataset.save_to_disk(self.combined_dataPath)

        # Save the arguments of the two task run for future reference.
        self.original_locs = self.data_locs
        self.original_args = self.data_args
        # Change the data sets and args to be the combined dataset 
        # and the third task dataset.
        self.data_locs = [pathExtention, data_loc]
        self.data_args = [None, data_arg]
        # Change to the new datanames.
        self.data_name0 = self.data_locs[0].split("/")[-1]
        self.data_name1 = self.data_locs[1].split("/")[-1]

        # From here we simply proceed like we are preforming a two
        # task FIP
        self.prep_model_datasets()
        # Define a metric to evaluate your models.
        self.metric = load_metric("accuracy")
        # Define a data collator that will keep a uniform input lenght.
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        # Check if we are using a checkpoint or if two task has already been run.
        if self.two_ct == None: #not in locals():
            assert two_ct_checkpoint != None, \
            "A two task FIP checkpoint must be passed\
             if two_task() has not been run"
            
            self.two_ct = AutoModelForSequenceClassification.from_pretrained(
                                        two_ct_checkpoint, num_labels=self.num_labels,
                                        ignore_mismatched_sizes=True)
        self.model = self.two_ct
        global model_ori
        model_ori = copy.deepcopy(self.model) #Model already trained on two tasks
        model_ori.to('cuda:0')
        # Save the data loader from a trainer based on the FIP run
        # it will be used in the custom trainer.
        global prevData_loader
        prevData_loader = self.get_prev_data_loader()
        self.additional_ct =  self.run_custom_trainer()
        return self.additional_ct

    '''
    This method will combine two datasets in DataDict format.
    The large dataset wil be subsampled to create and 50/50
    split between the two dataset. 
    dataset_1 will have its labels extended to avoid overlap.
    Both datasets will have ther number of classes and 
    class names combined.
    '''
    def combine_dataset(self, dataset_0, dataset_1):
        
        # Find the size of the smaller dataset and subsample the dataset
        # to that upper limit.
        min_test =  min(len(dataset_0['test']), len(dataset_1['test']))
        min_train =  min(len(dataset_0['train']), len(dataset_1['train']))
        dataset_0['test'] = dataset_0['test'].shuffle(seed=42).select(range(min_test))
        dataset_0['train'] = dataset_0['train'].shuffle(seed=42).select(range(min_train))
        dataset_1['test'] = dataset_1['test'].shuffle(seed=42).select(range(min_test))
        dataset_1['train'] = dataset_1['train'].shuffle(seed=42).select(range(min_train))
        
        # Extend the class lables so there is no overlap.
        class_label0 = dataset_0['test'].info.features['label']
        class_label1 = dataset_1['test'].info.features['label']
        self.label_len = class_label0.num_classes

        # Increase the range of lables in the both dataset to the total number of labels,
        # so they can be combined.
        total_classes = class_label1.num_classes + class_label0.num_classes
        dataset_1['test'].info.features['label'].num_classes = total_classes
        dataset_1['train'].info.features['label'].num_classes = total_classes
        dataset_0['train'].info.features['label'].num_classes = total_classes
        dataset_0['test'].info.features['label'].num_classes = total_classes
        # Add the both dataset names to both datasets name list to accomidate 
        # the large label number.
        all_class_names = class_label0.names + class_label1.names
        dataset_1['test'].info.features['label'].names =  all_class_names
        dataset_1['train'].info.features['label'].names = all_class_names
        dataset_0['test'].info.features['label'].names = all_class_names
        dataset_0['train'].info.features['label'].names = all_class_names

        # Increase the second datasets labels so there is no overlap with the first dataset.
        dataset_1['train'] = dataset_1['train'].map(self.update_label)
        dataset_1['test'] = dataset_1['test'].map(self.update_label)
        
        # Remove everything except the 'text' and 'label' from the 'train' and 'test' sets
        dataset_1 = self.clean_dataset(dataset_1)
        dataset_0 = self.clean_dataset(dataset_0)

        # Basic check to make sure they will concatenate (necessary but not sufficient)
        assert dataset_0['test'].features.type == dataset_1['test'].features.type
        
        # Combine the two two test and train datasets
        two_task_dataset_test = concatenate_datasets([dataset_0['test'], dataset_1['test']])
        two_task_dataset_train = concatenate_datasets([dataset_0['train'], dataset_1['train']])
        # Put the combined train and test set into one DatasteDict
        two_tast_dict = DatasetDict()
        two_tast_dict['test'] = two_task_dataset_test
        two_tast_dict['train'] = two_task_dataset_train

        self.two_task_dataset  = two_tast_dict
        
        return self.two_task_dataset
    # This method removes everything except the 'text' and 'label' from 
    # the 'train' and 'test' sets
    def clean_dataset(self, dataset):
        print('1',dataset['train'].features.keys())
        # pop everythign that isn't "train" or "test" (E.g. "validation")
        columns = list(dataset.keys())
        pop_list = [col for col in columns if col not in ["train", "test"]]
        for pop in pop_list:
            dataset.pop(pop)

        # Change the name of the text to 'text' incase it is anything else.
        features =list(dataset['train'].features.keys())
        print('2',features)
        if 'label' in features:
            features.remove('label')
        key = features[0]
        if key != "text":
             dataset = dataset.rename_column(key, "text")
        # Remove everything but the 'text' and the 'label' for uniformity.
        column_names = [col for col in features if col not in ["text", "label"]]
        print(column_names)
        dataset = dataset.remove_columns(column_names)
        print(dataset['train'].features.keys())
        return dataset
    
    # Prepare the two datasets and run all three trainers with one command.
    def two_task(self):
        # Load the datasets and extend the labeling to remove overlap.
        self.prep_model_datasets()
        # Define a metric to evaluate your models.
        self.metric = load_metric("accuracy")
        # Define a data collator that will keep a uniform input lenght.
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        # Train the saved model on the first dataset.
        self.model = self.train_first_dataset()
        # Save the data loader from a trainer based on the first dataset
        # it will be used in the custom trainer.
        global prevData_loader
        prevData_loader = self.get_prev_data_loader()
        self.two_ct_trainer = self.run_custom_trainer()
        return  self.two_ct_trainer.modle
         
        
    
    def prep_model_datasets(self):
        download_model(self)
    
        # Load both datasets and extend the labeling of the second to eliminate overlap.
        self.load_datasets()
        self.load_extended_labels()

        self.num_labels = (len(np.unique(self.datasets[0]['train']['label'])) + 
                            len(np.unique(self.datasets[1]['train']['label'])))
        print("Total # of labels = ", self.num_labels)
        # Load the model provided by the user with the correct number labels based on the two dataset.
        self.hf_model = AutoModelForSequenceClassification.from_pretrained(self.HF_loc, num_labels=self.num_labels)


    # Tokenize a Dataset.
    def tokenize_data(self, dataset, dataPath):
        tokenDataPath = dataPath +'_token'
        # Check if this dataset has been saved and tokenized before.
        if os.path.exists(tokenDataPath):
            tokenized_datasets = load_from_disk(tokenDataPath)
        # If not tokenize it and save it.
        else:
            features =list(dataset['train'].features.keys())
            print(features)
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
        
    #Load and tokenize both datasets.
    def load_datasets(self):
        print("Load datasets")
        # Load the first dataset save its data path and tokenize it.
        self.dataset_0 = ld(self, 0)
        self.dataPath_0 = self.dataPath
        self.token_data_0 = self.tokenize_data(self.dataset_0, self.dataPath_0)
        
        # Load the second dataset save its data path and tokenize it. 
        self.dataset_1 = ld(self, 1)
        self.dataPath_1 = self.dataPath
        self.token_data_1 = self.tokenize_data(self.dataset_1, self.dataPath_1)

        # If we are in short mode we will shrink the dataset to tiny subset.
        if self.short:
            self.token_data_0['test'] = self.token_data_0['test'].shuffle(seed=42).select(range(2000))
            self.token_data_0['train'] = self.token_data_0['train'].shuffle(seed=42).select(range(5000))
            self.token_data_1['test'] = self.token_data_1['test'].shuffle(seed=42).select(range(2000))
            self.token_data_1['train'] = self.token_data_1['train'].shuffle(seed=42).select(range(5000))

        # Put both dataset, data paths, and the tokenized datasets into lists.
        self.datasets = [self.dataset_0, self.dataset_1]
        self.dataPaths = [self.dataPath_0, self.dataPath_1]
        self.token_datasets = [self.token_data_0, self.token_data_1]
    
    # Extened the labeling so there is no overlap in classes between datasets.
    def update_label(self, example):
        example['label'] = example['label'] + self.label_len
        return example                     

    def extend_class(self, dataset_0, dataset_1):
        # Extend the class lables so there is no overlap.
        class_label0 = dataset_0['test'].info.features['label']
        class_label1 = dataset_1['test'].info.features['label']
        # Increase the number of lables in the second dataset to the total number of label.
        class_label1.num_classes = class_label1.num_classes + class_label0.num_classes
        dataset_1['train'].info.features['label'].num_classes = class_label1.num_classes
        # Add the first dataset names to second datasets name list to accomidate the large label number.
        dataset_1['test'].info.features['label'].names = class_label0.names + class_label1.names 
        dataset_1['train'].info.features['label'].names = class_label1.names
        # Increase the second datasets labels so there is no overlap with the first dataset.
        self.label_len = class_label0.num_classes
        dataset_1['train'] = dataset_1['train'].map(self.update_label)
        dataset_1['test'] = dataset_1['test'].map(self.update_label)

        return dataset_1

    # Apply te label extention and save the tokenized datasets.
    def load_extended_labels(self):
        self.label_len = len(np.unique(self.datasets[0]['train']['label']))
        extneded_path = self.dataPaths[1]+'_token_extended'
        # Check if the extended version has created and saved before.
        if os.path.exists(extneded_path):
            self.token_datasets[1] = load_from_disk(extneded_path)
        else:
            self.token_datasets[1] = self.extend_class(self.token_datasets[0], self.token_datasets[1])

            #class_label0 = self.token_datasets[0]['test'].info.features['label']
            #class_label1 = self.token_datasets[1]['test'].info.features['label']
            ## Increase the number of lables in the second dataset to the total number of label.
            #class_label1.num_classes = class_label1.num_classes + class_label0.num_classes
            #self.token_datasets[1]['train'].info.features['label'].num_classes = class_label1.num_classes
            ## Add the first dataset names to second datasets name list to accomidate the large label number.
            #self.token_datasets[1]['test'].info.features['label'].names = class_label0.names + class_label1.names 
            #self.token_datasets[1]['train'].info.features['label'].names = class_label1.names
            ## Increase the second datasets labels so there is no overlap with the first dataset.
            #self.token_datasets[1]['train'] = self.token_datasets[1]['train'].map(self.update_label)
            #self.token_datasets[1]['test'] = self.token_datasets[1]['test'].map(self.update_label)

            self.token_datasets[1].save_to_disk(self.dataPaths[1]+'_token_extended')            
                
    
    # This method is used by the trainer to evaluate the model.
    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return self.metric.compute(predictions=predictions, references=labels) 
    
    # This method trains the first dataset on passed model.
    def train_first_dataset(self):
        print("Train First Model")
        # Grab the name of the model from the model location.

        # Create a location to save tainer checkpoint.
        full_name = f"{self.model_name}-finetune-{self.data_name0}"
        output_path =  self.save_loc +'/trainer_checkpoint/'
        output_path += full_name
        
        # Define training args.
        training_args = TrainingArguments(
            output_dir=output_path,
            evaluation_strategy = "steps",
            eval_steps = 100, # Evaluation and Save happens every 100 steps.
            save_total_limit = 5, # Only last 5 models are saved. Older ones are deleted.
            learning_rate=2e-5,
            weight_decay=0.1,#0.01
            push_to_hub=False,
            num_train_epochs=self.num_train_epochs,#args.num_train_epochs,
            per_device_train_batch_size=8,
            metric_for_best_model = 'accuracy',
            load_best_model_at_end=True
        )

        # Check if it exists and load it if it exists.
        if os.path.exists(output_path+'/*'):
            check_points = os.listdir(output_path)
            print(len(check_points))
            last_cp = output_path+'/'+check_points[-1]
            trainer = AutoModelForSequenceClassification.from_pretrained(last_cp, num_labels=self.num_labels)
        else:
        # TRAIN NETWORK ON FIRST DATASET
            self.trainer = Trainer(
                        model=self.hf_model,
                        args=training_args,
                        train_dataset=self.token_datasets[0]['train'],
                        eval_dataset=self.token_datasets[0]['test'],
                        data_collator=self.data_collator,
                        tokenizer=self.tokenizer,
                        compute_metrics=self.compute_metrics,
                        callbacks = [EarlyStoppingCallback(early_stopping_patience=0)]
                        )

            self.trainer.train()
        # Save the trained model as the 'original'.
        self.model = self.trainer.model
        global model_ori
        model_ori = copy.deepcopy(self.model) #Model already trained on yelp
        model_ori.to('cuda:0')
        #model_ori = self.copy_ori(self.model,'originals/'+full_name)
        return self.model   

    # Define a trainer which will only be used to extract the dataloader
    # and will not be trained.
    def get_prev_data_loader(self):
        print("Get Previous Data Loader")
        self.data_name1 = self.data_locs[1].split("/")[-1]
        full_name = f"{self.model_name}-finetune-{self.data_name1}"
        output_path =  self.save_loc +'/trainer_checkpoint/'
        output_path += full_name
        # Define the traing arguments 
        training_args = TrainingArguments(
            output_dir=output_path,
            evaluation_strategy = "no",
            save_total_limit = 5,
            save_steps=50,
            learning_rate=2e-5,
            weight_decay=0.01,
            push_to_hub=False,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=8   
        )

        #Subsample small number of examples from the yelp dataset.
        small_train_first = self.token_datasets[0]['train'].shuffle(seed=42).select(range(2000))
        small_test_first = self.token_datasets[0]['test'].shuffle(seed=42).select(range(1500))

        print("First small dataset", small_test_first)


        # Define the trainer which will only be used for the Dataloader
        trainer1 = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=small_train_first,
            eval_dataset=small_test_first,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
            )
        # Assign the dataloader to the class instance
        self.prevData_loader = trainer1.get_train_dataloader()
        return self.prevData_loader
    
    '''
     Run the custom trainer on both datasets.
     This method requires both datasets to be tokenized
     and prevData_loader and model_ori must be defined from get_prev_data_loader()
     and train_first_dataset() respectively. 
    '''
    def run_custom_trainer(self):
        print('Run Custom Trainer')
        full_name = f"{self.model_name}-finetune-{self.data_name0}-{self.data_name1}"
        output_path =  self.save_loc +'/trainer_checkpoint/'
        output_path += full_name
        # Define the training arguments.
        self.custom_training_args = TrainingArguments(
            output_dir=output_path,
            evaluation_strategy = "no",
            save_total_limit = 5,
            save_steps=500,
            learning_rate=2e-5,
            weight_decay=0.01,
            push_to_hub=False,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=8   
        )
        
        # Initalize the custom trainer with the second dataset.
        self.custom_trainer = CustomTrainer(
            model=self.model,
            args=self.custom_training_args,
            train_dataset=self.token_datasets[1]['train'],
            eval_dataset=self.token_datasets[1]['test'],
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
            )
        # Pass the data loader and original model to the custom trainer rather than
        # using gloabl variables. 

        #self.custom_trainer.prevData_loader = prevData_loader
        #self.custom_trainer.prevData_loader = self.model_ori
        print("CUSTOM TRAINING") 
        self.custom_trainer.train()
        return self.custom_trainer

                                
            