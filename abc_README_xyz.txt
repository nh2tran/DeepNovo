DeepNovo is implemented and tested with Python 2.7 and TensorFlow r0.10. In addition, DeepNovo also uses Numpy and Cython.
To run ALPS, you have to install Java Runtime verison 7 or the newer version. Please visit http://java.com/en/download/ to download and install the lastest version.

How to use DeepNovo and ALPS to do antibody sequence assembly?

   Step 1: De novo antibody peptide sequencing.
        
        Currently DeepNovo supports training and testing modes. Hence, the real peptides need to be provided in the input mgf files with tag "SEQ=". If you want to do de novo antibody sequencing, you can provide any arbitraty sequence for tag "SEQ=" to bypass the reading procedure. In the output file, you can ignore the calculation of accuracy and simply use the predicted peptide sequence.
            
        All other options can be found in "data_utils.py".

        In this antibody project, only one pre-trained model was used and located in the following folder:
        train.doremon.resolution_50.epoch_20.da_4500.ab.training.mouse 
        
        The de novo sequencing mgf file is defined in "data_utils.py", for example:
            decode_test_file = "data/ab.testing/assem.waters.mouse.heavy/peaks.refine.mgf"
            decode_test_file = "data/ab.testing/assem.waters.mouse.light/peaks.refine.mgf"
    
        Finally, accomplish de novo antibody sequencing by running the following command:
        python main.py --train_dir train.doremon.resolution_50.epoch_20.da_4500.ab.training.mouse --decode --beam_search --beam_size 10
         
        The results are written to "decode_output.tab" in the model folder.
        
    
    Step 2: Extract information from decode_output.tab.
        
        Rename and remove "decode_output.tab" to destination folder, for example:
        rename "decode_output.tab" as "waters.heavy.da_4500.decode_output.tab" and remove it to folder data/ab.testing/assem.waters.mouse.heavy/
        
        In "testing_utils.py", run the following command to acquire the extracted file "deepnovo_totalscore.csv":
        deepnovo_to_ALPS(deepnovo_file="waters.mouse.light.da_4500.decode_output.tab",output_file="data/ab.testing/assem.waters.mouse.light/deepnovo_totalscore.csv")
            
    
    Step 3: Remove sequence contaminants from de novo sequencing result (deepnovo_totalscore.csv) and retrieve the filtered antibody peptides based on a percentage.
        
        Here, we set percentage as 50%, and get the final file "data/ab.testing/assem.waters.mouse.heavy/deepnovo_filter_0.5.csv" and "data/ab.testing/assem.waters.mouse.light/deepnovo_filter_0.5.csv".
        All the filtered peptides are stored in "data/ab.testing/assem.waters.mouse.heavy/deepnovo_filter.csv" and "data/ab.testing/assem.waters.mouse.light/deepnovo_filter.csv", respectively.
    
    
    Step 4: Run ALPS to assemble a antibody sequence.
    
        After install Java in your computer, use the following command to check the usage of ALPS:
        
        java -jar ALPS.jar
        
        The usage will be prompted as following:
        
        Usage:
        
       			ALPS <input_file> <k> [<c>]
                		<input_file>: .csv files containing input data;
                		<k>         : the length of k-mers, 6 or 7 is recommended.
                    <c>         : the number of top contigs to use for assembly, default is 10.	

        Run ALPS with the required parameters.
        
        Example: java -jar ALPS.jar data\assembledata_WIgG1_Heavy\deepnovo_filter_0.5.csv 6
        
    
    
This is an on-going project, hence the code is quite messy. Please feel free to send any questions/suggestions to x322zhan@uwaterloo.ca.

Cheers!
Xianglilan Zhang (PhD)
School of Computer Science, University of Waterloo, Canada.
                  
  
