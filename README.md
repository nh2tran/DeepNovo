# DeepNovo

## Protein Identification with Deep Learning: from abc to xyz.

DeepNovo is a deep learning-based tool to address the problem of protein 
identification from tandem mass spectrometry data. The core model applies 
convolutional neural networks and recurrent neural networks to predict the
amino acid sequence of a peptide from its spectrum, a similar task to
generating a caption from an image. We combine two techniques, de novo sequencing 
and database search, into a single deep learning framework for peptide identification, 
and use de Bruijn graph to assemble peptides into protein sequences.

More details are available in our publications: 

- Protein identification with deep learning: from abc to xyz. *arXiv (to be updated)*.

- De novo peptide sequencing by deep learning. *Proceedings of the National Academy of Sciences, 2017*.

**If you want to use the models in our PNAS paper, please use the branch PNAS**.

## Update version 0.0.1

We have added the database search function into DeepNovo. Both modules de novo 
sequencing and database search are now available.

The pre-trained model, training and testing data can be downloaded from here:

https://drive.google.com/open?id=0By9IxqHK5MdWalJLSGliWW1RY2c

The following updates are also included in this version: 

- The implementation has been upgraded and tested on TensorFlow 1.2.

- The code has been cleaned up with PEP8 and TensorFlow pylint guides, but many 
docstrings are still to be added.

- Functional modules including I/O, training, de novo sequencing, database search, 
and testing should be group into separate worker classes. Same for the neural 
network models. 

