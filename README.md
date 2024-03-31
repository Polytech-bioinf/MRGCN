# MRGCN
Code of MRGCN:Cancer subtyping with multi-reconstruction gragh convolutional network using full and partial multi-omics dataset. MRGCN is a cancer subtype framework based on multi-omics profiles. The input for the framework is mRNA, miRNA and DNA methylation. The output is the consensus representation shared by all omics. 


train consensus representation
```
#networks.py is used to learn consensus representation
#the models are saved in train_models
```

MRGCN is mainly divided into two components: 1. networks is used to learn consensus representation from all omics. 2. Consensus clustering determine the number of subtypes and the cluster label corresponding to each sample. 


the program of the comparison methods :
 ```
 ./enrichment_test
 ```

MRGCN is based on the Python program language and R language. The network's implementation was based on the open-source library Pytorch 1.9.0. 
