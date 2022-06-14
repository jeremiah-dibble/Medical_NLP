import datasets
import numpy as np
from datetime import datetime
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, Trainer)

###EXAMPLE###
# model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)
# dataset = datasets.load_from_disk('datasets/yelp_review_full')
# tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")short_dataset =datasets.load_from_disk('datasets/yelp_review_full')
# short_dataset["test"]=dataset["test"].shuffle(seed=1).select(range(1000))
# import model_test as mt
# mt.test_throughput(model, dataset['test'], tokenizer)
#####################

def test_throughput(model, dataset, tokenizer):
    test_trainer = Trainer(model) 
    
    start = datetime.now()
    
    token_dataset = tokenize_troughput(dataset, tokenizer)
    raw_pred, _, _ = test_trainer.predict(token_dataset) 
    y_pred = np.argmax(raw_pred, axis=1)
    
    stop = datetime.now()
    seconds = (stop-start)
    rate = seconds/len(dataset)
    print("This file took: ", seconds)
    print('At a rate of ' + str(rate) +' per line')
    return y_pred, rate
    
def tokenize_troughput(dataset, tokenizer):
    features =list(dataset.features.keys())
    if 'label' in features:
        features.remove('label')
    key = features[0]
    tokenized_datasets = dataset.map(lambda examples: tokenizer(examples[key],
                                                                               max_length=512, 
                                                                            padding="max_length", 
                                                                               truncation=True),
                                                                                batched=True)
    return tokenized_datasets