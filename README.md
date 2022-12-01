# Fake_news_detection Group 38
## Amulya Sethurao, Harshith Acharya, Mohammed Zubair Khan, Purva Makarand Kulkarni, Vinya Somayajula

## Abstract:
Fake news is considered to be one of the greatest threats to commerce, journalism and democracy all over the world and with huge collateral damages. In our project, we use the LIAR dataset of real world political statements to identify if it as fake or not. Our project not only focuses on the content but on the subject, the source and the title as well to formulate both binary and 6 class classification problem with labels as true, mostly true, partially true, partially false, mostly false and false.

## How to run
Run main.py which is the driver of the experiments. To train a model change the variable mode in main.py to train. For evaluating a saved model, change mode to test and put the name of the saved model in the variable pathModel. To run LIAR dataset, change the variable dataset_name to LIAR and if you want to run LIAR-PLUS dataset then change dataset_name to LIAR-PLUS.
Currently main.py gives binary results.


To run BERT model, run BERT_DAPT_LIAR.py for training and test.py for testing. For running test.py checkpoint path is needed. We tested on the bert_model_test_roberta_binary.pth checkpoint for RoBERTa model with domain adaptive pretraining. 


## Results

| Model         | Accuracy      | metric|
| ------------- |:-------------:|:-----:|
| Baseline for CNN with attention & BiLSTM  | 0.63    | 0.24 |
| Baseline with BERT| 0.75      |   0.32 |
|CNN with attention & GRU | 0.61	| 0.25 |
| BERT with domain adaptive  pre-fine tuning | 0.75 | 0.34 |







## References:
1. https://github.com/manideep2510/siamese-BERT-fake-news-detection-LIAR
2. https://github.com/ekagra-ranjan/fake-news-detection-LIAR-pytorch
3. https://huggingface.co/allenai/news_roberta_base
4. https://www.aclweb.org/anthology/P17-2067.pdf

