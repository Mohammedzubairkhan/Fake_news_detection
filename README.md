# Fake_news_detection

Current accuracy 
LIAR Test Accuracy: 0.622
LIAR-PLUS Test Accuracy: 0.633


1. Change Bi-LSTMs to GRU units [Zubair started this]
2. Implement Hierarchical network 
3. Fine tuning 

Run main.py which is the driver of the experiments. To train a model change the variable mode in main.py to train. For evaluating a saved model, change mode to test and put the name of the saved model in the variable pathModel. To run LIAR dataset, change the variable dataset_name to LIAR and if you want to run LIAR-PLUS dataset then change dataset_name to LIAR-PLUS.
Currently main.py gives binary results.
