#!/usr/bin/env python
'''
# /////////////////////////////////////////////////////////////////////////////////////////////////////
#                            MLPNN Optical Character Recognition Demonstration
#
#     Engineers: Josh Sackos
#                jsackos@pdx.edu
#                503.298.1820
#
#                Khyati Sinha
#                ksinha@pdx.edu
#
#       School:  Portland State University
#       Course:  ECE510-002 Embedded Vision 1, Fall 2017
#      Project:  MLPNN Optical Character Recognition
#  Description:  A Python progrma for creating an OCR MLPNN neural network via the MlpnnOcr class.
#
#     SVN Info:
#                $Author: jsackos $
#                $Date: 2017-12-02 20:11:50 -0800 (Sat, 02 Dec 2017) $
#                $Revision: 69 $
#                $Id: main.py 69 2017-12-03 04:11:50Z jsackos $
#                $Header: https://jsackos@projects.cecs.pdx.edu/svn/jsackos-mlpnnocr/python/main.py 69 2017-12-03 04:11:50Z jsackos $
#                $HeadURL: https://jsackos@projects.cecs.pdx.edu/svn/jsackos-mlpnnocr/python/main.py $
#
# /////////////////////////////////////////////////////////////////////////////////////////////////////
'''

# /////////////////////////////////////////////////////////////////////////////////////////////////////
#                                               Imports
# /////////////////////////////////////////////////////////////////////////////////////////////////////
import sys, os, time, random
import numpy as np
import sklearn
import cv2
from mlpnnocr import MlpnnOcr

# /////////////////////////////////////////////////////////////////////////////////////////////////////
#                                        Configuration Variables
# /////////////////////////////////////////////////////////////////////////////////////////////////////

# >>>>> Training/Testing Configuration <<<<<
trainPercent       = 70.0;                           # Percentage of samples to use for training
nSamples           = 20000;                          # Total #of training/testing images to generated
maxCharLength      = 4;                              # Maximum number of characters in a word
maxWordLength      = 3;                              # Maximum number of words in a generated image
data_directory     = '../data';                      # Path to where training/testing images are
save_path          = '../model/mlpnnocr_model.pkl';  # File to save trained neural network results to
# Possible characters
alpha_chars        = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','&','@'];

# >>>>> Neural Network Configuration <<<<<
nnConfig = {
            'solver'            : 'sgd'     ,        # Solver for weight optimization
            'activation'        : 'relu'    ,        # Activation method to use for neurons
            'hidden_layer_sizes': (80, 40)  ,        # Neural network hidden layers and #of neurons in each       # Good (80,40)
            'learning_rate'     : 'constant',        # Type of learning rate, constant, adaptive, and invscaling
            'learning_rate_init': 0.001     ,        # Initial learning rate for stochastic gradient descent
            'max_iter'          : 100       ,        # Number of epochs to train for, for 'sgd'
            'momentum'          : 0.9       ,        # Helps keep weights moving in the same direction during training
            'shuffle'           : False              # Vectors already get randomly shuffled in buildTrainTestArrays
};

'''
# //////////////////////////////////////////////////////////////////////////////////////////////////
#                                               main()
#
#   Description:  Main program to run for creating an OCR neural network, for the MLPNN Optical
#                 Character Recognition project.
#     
#     Arguments:  N/A
#       
#       Returns:  N/A
#          
# //////////////////////////////////////////////////////////////////////////////////////////////////
'''
def main():

   # >>>>> Create an instance of MlpnnOcr class <<<<<
   mlpnnocr_inst = MlpnnOcr(alpha_chars, data_directory);
   
   # >>>>> Generate Training/Testing Data If It Does Not Exist <<<<<
   if not os.path.exists(data_directory):
      mlpnnocr_inst.genTrainingTestingData(nSamples, 1, 1);

   # >>>>> Import Training/Testing Data <<<<<
   mlpnnocr_inst.importTrainingTestingData(trainPercent);

   # >>>>> Build training/testing arrays <<<<<
   mlpnnocr_inst.buildTrainTestArrays();

   # >>>>> Create Neural Network <<<<<
   mlpnnocr_inst.createNN(nnConfig);
   
   # >>>>> Train Neural Network <<<<<
   mlpnnocr_inst.trainNN(save_path);

   # >>>>> Test Neural Network <<<<<
   mlpnnocr_inst.testNN();

   # >>>>> Program Finished! <<<<<
   print("Done!");

# /////////////////////////////////////////////////////////////////////////////////////////////////////
#                                      Application Entry Point
# /////////////////////////////////////////////////////////////////////////////////////////////////////
if __name__ == "__main__":

   main();
   



   

   
      
      















