# DeepNovo

## Latest update: De novo sequencing for both DDA and DIA.

- Moving forward, we use a feature-based framework to unify DDA and DIA data analysis. As the data and model structure change, and to keep this DDA repository intact, we will maintain the new framework in a different repository: https://github.com/nh2tran/DeepNovo-DIA.

- Publication: Deep learning enables de novo peptide sequencing from DIA mass spectrometry. Nature Methods, 2018. (https://www.nature.com/articles/s41592-018-0260-3) 

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

- The first-ever hybrid tool for peptide identification that performs de novo
sequencing and database search under the same scoring and sequencing framework.
DeepNovo now have three sequencing modes: `search_denovo()`, `search_db()`, and 
`search_hybrid()`.

- Added decoy database search to estimate False Discovery Rate (FDR). The FDR
can be used to filter both database search and de novo sequencing results.

- Replaced DecodingModel by ModelInference to make the code of building neural
network models easy to understand and for further development.

We have decided to still use low-level functions of TensorFlow to construct
neural networks. We think they could help to get better understanding of the
basic details of our model and how to improve it. The network architecture is
not so complicated, so the code is not too messy even with low-level functions.
We will eventually update with high-level ones such as tf.layers and others.

We have added the database search function into DeepNovo. Both modules de novo 
sequencing and database search are now available.

The pre-trained model, training and testing data can be downloaded from here:

https://drive.google.com/drive/folders/1qB8wDBnnm1qw0wDuSCxOoxkyV-b4LkTo?usp=sharing

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
    
