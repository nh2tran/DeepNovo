DeepNovo is implemented and tested with Python 2.7 and TensorFlow r0.10. In addition, DeepNovo also uses Numpy and Cython.

How to use DeepNovo?

    Step 0: Build cython_speedup to accelerate Python with C.
    
        python setup.py build_ext --inplace
        
    Step 1: Test a pre-trained DeepNovo using the following command.
    
        python main.py --train_dir train.deepnovo.high.cross.9high_80k.exclude_bacillus --decode --beam_search --beam_size 10
        
        There are nine pre-trained high-resolution models located in the following folders:
            train.deepnovo.high.cross.9high_80k.exclude_bacillus
            train.deepnovo.high.cross.9high_80k.exclude_clambacteria
            train.deepnovo.high.cross.9high_80k.exclude_honeybee
            train.deepnovo.high.cross.9high_80k.exclude_human
            train.deepnovo.high.cross.9high_80k.exclude_mmazei
            train.deepnovo.high.cross.9high_80k.exclude_mouse
            train.deepnovo.high.cross.9high_80k.exclude_ricebean
            train.deepnovo.high.cross.9high_80k.exclude_tomato
            train.deepnovo.high.cross.9high_80k.exclude_yeast

        The testing mgf file is defined in "data_utils.py", for example:
            decode_test_file = "data/high.bacillus.PXD004565/peaks.db.10k.mgf"
            decode_test_file = "data/high.bacillus.PXD004565/peaks.db.mgf"
        
        The results are written to "decode_output.tab" in the model folder.
            
    Step 2: Train a DeepNovo model using the following command.
      
        python main.py --train_dir train.example --train

        The training mgf files are defined in "data_utils.py", for example:
            data_format = "mgf"
            input_file_train = "data/cross.9high_80k.exclude_bacillus/cross.cat.mgf.train.repeat"
            input_file_valid = "data/cross.9high_80k.exclude_bacillus/cross.cat.mgf.valid.repeat"
            input_file_test = "data/cross.9high_80k.exclude_bacillus/cross.cat.mgf.test.repeat"
        
        The model files are written to the training folder "train.example"
        
    Step 3: De novo sequencing.
    
        Currently DeepNovo supports training and testing modes. Hence, the real peptides need to be provided in the input mgf files with tag "SEQ=". If you want to do de novo sequencing, you can provide any arbitraty sequence for tag "SEQ=" to bypass the reading procedure. In the output file, you can ignore the calculation of accuracy and simply use the predicted peptide sequence.
            
    All other options can be found in "data_utils.py".
    
This is an on-going project, hence the code is quite messy. Please feel free to send any questions/suggestions to nh2tran@uwaterloo.ca.

Cheers!
Hieu Tran (PhD)
School of Computer Science, University of Waterloo, Canada.
Email: nh2tran@uwaterloo.ca
                  
  
