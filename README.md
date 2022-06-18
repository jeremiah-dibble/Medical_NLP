# Medical_NLP
A repository for work done by Jeremiah Dibble for Matt Thomson's Lab at Caltech

Bio_encoder.ipynb takes a csv of patient information and encodes each line using a huggingface model.

Fine_Tune_ICD.ipynb fine tunes a huggingface model to predict iCD9 codes based on the procedures description in the patient record.

ft.py defines a class that will finetune a huggingface model on any csv or huggingface dataset that contain a coulmn containg text and 'label' column.
      Additionaly it can be used to mask and tokenize datasets.

FIP_paperspace.py defines a class that will implement the FIP method using a huggingface model and two huggingface datasets
