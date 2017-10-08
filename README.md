# DeepNovo

## Protein Identification with Deep Learning: from abc to xyz.

## Update version 0.0.1

We have added the database search function into DeepNovo. Our goal is to combine 
two modules de novo sequencing and database search into a single deep learning 
framework for peptide identification. Furthermore, de Bruijn graph assembly 
technique is integrated to offer a complete solution to the problem of protein 
identification from tandem mass spectrometry data. 

The pre-trained model, training and testing data can be downloaded from here:

https://drive.google.com/open?id=0By9IxqHK5MdWalJLSGliWW1RY2c

The following updates are also included in this version: 

- The implementation has been upgraded and tested on TensorFlow 1.2.

- The code has been cleaned up with PEP8 and TensorFlow pylint guides, but many 
docstrings are still to be added.

- Functional modules including I/O, training, de novo sequencing, database search, 
and testing should be group into separate worker classes. Same for the neural network models. 

*If you want to use the models in our PNAS paper, please use the branch PNAS and 
instructions of the project below.*

## Project: De novo Peptide Sequencing by Deep Learning.

Publication: De novo Peptide Sequencing by Deep Learning. *Proceedings of the National Academy of Sciences, 2017*.

Data repository: ftp://massive.ucsd.edu/MSV000081382

##

DeepNovo is implemented and tested with Python 2.7 and TensorFlow r0.10. In addition, DeepNovo also uses Numpy and Cython.

Here we provide instructions for installing Python and TensorFlow on Linux Ubuntu.
    
    Step 1: Install Python 2.7, pip, and virtualenv using the following commands
    
        $ sudo apt-get install python2.7
        $ sudo apt-get install python-pip python-dev python-virtualenv
        
    You can then check the version of Python using the following command
    
        $ python --version
        
    Step 2: Install TensorFlow with virtualenv.

      Virtualenv is a virtual Python environment isolated from other Python development, incapable of interfering with or being affected by other Python programs on the same machine. 
      To start working with TensorFlow, you simply need to "activate" the virtual environment. 
      When you are done using TensorFlow, you may "deactivate" the environment.
      
      Step 2.1: Create a virtualenv environment using the following command
      
        $ virtualenv --system-site-packages ~/tensorflow

      Step 2.2: Activate the virtualenv environment using the following command
      
        $ source ~/tensorflow/bin/activate
        
      Step 2.3: Install TensorFlow version r.0.10 using the following command where URL depends on CPU/GPU support
      
        (tensorflow)$ pip install --upgrade URL 

        # with CPU support only
        URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.10.0-cp27-none-linux_x86_64.whl

        # with NVIDIA-GPU support
        URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.10.0-cp27-none-linux_x86_64.whl

      Step 2.4: Check TensorFlow version, installation, device
      
        $ python
        >>> import tensorflow as tf
        >>> print(tf__version__)
        >>> sess = tf.InteractiveSession()
        >>> sess.close()
        >>> quit()

      Step 2.5: Deactivate the virtualenv environment using the following command
      
        (tensorflow)$ deactivate
        
    Step 3: Install Cython using the following command
        
        $ pip install Cython
        
To use DeepNovo in the TensorFlow enviroment, you need to activate/deactivate the enviroment as mentioned earlier. Further details of DeepNovo usage are provided in the README files in each folder.
