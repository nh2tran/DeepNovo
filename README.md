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

- Protein identification with deep learning: from abc to xyz. *arXiv:1710.02765, 2017*.

- De novo peptide sequencing by deep learning. *Proceedings of the National Academy of Sciences, 2017*.

- Complete de novo assembly of monoclonal antibody sequences. *Scientific Reports, 2016*.

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

## How to use DeepNovo?

DeepNovo is implemented and tested with Python 2.7, TensorFlow 1.2 and Cython.

**Step 0**: Build deepnovo_cython_setup to accelerate Python with C.

    python deepnovo_cython_setup.py build_ext --inplace

**Step 1**: Test a pre-trained model with DeepNovo de novo sequencing

    python deepnovo_main.py --train_dir train.example --decode --beam_search --beam_size 5

The testing mgf file is defined in "deepnovo_config.py", for example:

    decode_test_file = "data.training/yeast.low.coon_2013/peaks.db.mgf.test.dup"

**Step 2**: Test a pre-trained model with DeepNovo database search

    python deepnovo_main.py --train_dir train.example --search_db

The testing mgf file is defined in "deepnovo_config.py", for example:

    input_file = "data.training/yeast.low.coon_2013/peaks.db.mgf.test.dup"
        
The results are written to the model folder "train.example".

**Step 3**: Train a DeepNovo model using the following command.

    python deepnovo_main.py --train_dir train.example --train

The training mgf files are defined in "deepnovo_config.py", for example:

    input_file_train = "data.training/yeast.low.coon_2013/peaks.db.mgf.train.dup"

    input_file_valid = "data.training/yeast.low.coon_2013/peaks.db.mgf.valid.dup"

    input_file_test = "data.training/yeast.low.coon_2013/peaks.db.mgf.test.dup"

The model files will be written to the training folder "train.example".

**Step 4**: De novo sequencing.

Currently DeepNovo supports training and testing modes. Hence, the real peptides 
need to be provided in the input mgf files with tag "SEQ=". If you want to do 
de novo sequencing, you can provide any arbitraty sequence for tag "SEQ=" to 
bypass the reading procedure. In the output file, you can ignore the calculation 
of accuracy and simply use the predicted peptide sequence.

All other options can be found in "deepnovo_config.py".
    
