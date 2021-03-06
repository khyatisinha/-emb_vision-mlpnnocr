#!/usr/bin/env python
'''
# /////////////////////////////////////////////////////////////////////////////////////////////////////
#                            MLPNN Optical Character Recognition Demonstration
#
#     Engineers: Khyati Sinha
#                ksinha@pdx.edu
#
#                Josh Sackos
#                jsackos@pdx.edu
#                503.298.1820
#
#       School:  Portland State University
#       Course:  ECE510-002 Embedded Vision 1, Fall 2017
#      Project:  MLPNN Optical Character Recognition
#  Description:  
#
#     SVN Info:
#                $Author$
#                $Date$
#                $Revision$
#                $Id$
#                $Header$
#                $HeadURL$
#
# /////////////////////////////////////////////////////////////////////////////////////////////////////
'''

# /////////////////////////////////////////////////////////////////////////////////////////////////////
#                                           Global Variables
# /////////////////////////////////////////////////////////////////////////////////////////////////////

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
nSamples           = 10000;                           # Total #of training/testing images to generated
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

# >>>>> Demonstration Configuration <<<<<
demoConfig = {
              'n_demo_samples'    : 2                ,   # Only needed for Windows, to prevent sklearn fortran from crashing command prompt.
              'demo_period'       : 5.0               ,   # The amount of time between examples during demonstration
              'camera_pos'        : -1                ,   # Camera operating system position to use, if -1, no camera
              'resolution'        : (640,280)         ,   # Camera resolution, if used in demo
              'max_char'          : 4                 ,   # Maximum number of characters in a word
              'max_words'         : 3                 ,   # Maximum number of words in demo examples
              'low_thresh'        : 120               ,   # Low threshold value for cv2.threshold();
              'high_thresh'       : 255               ,    # High threshold value for cv2.threshold();
              'make_video'        : False             ,
              'video_resolution'  : (1220,460)        ,   # Camera resolution, if used in demo
              'save_path'         : 'demo_random.avi' ,
              'fps'               : 5.0
};

'''
# //////////////////////////////////////////////////////////////////////////////////////////////////
#                                               main()
#
#   Description:  Main program to run for demonstrating the MLPNN Optical Character Recognition
#                 project.
#     
#     Arguments:  N/A
#       
#       Returns:  N/A
#          
# //////////////////////////////////////////////////////////////////////////////////////////////////
'''
def main():

   # >>>>> Create an instance of our class <<<<<
   mlpnnocr_inst = MlpnnOcr(alpha_chars, data_directory);
   
   # >>>>> Generate Training/Testing Data If It Does Not Exist <<<<<
   if not os.path.exists(data_directory):
      mlpnnocr_inst.genTrainingTestingData(nSamples, 1, 1);

   # >>>>> Do stuff with it <<<<<
   mlpnnocr_inst.projectInfo();

   # >>>>> Import Training/Testing Data <<<<<
   mlpnnocr_inst.importTrainingTestingData(trainPercent);

   # >>>>> Build training/testing arrays <<<<<
   mlpnnocr_inst.buildTrainTestArrays();

#   # >>>>> Create Neural Network <<<<<
#   mlpnnocr_inst.createNN(nnConfig);

   # >>>>> Train Neural Network <<<<<
#   mlpnnocr_inst.trainNN(save_path);

   # >>>>> Create Neural Network <<<<<
   mlpnnocr_inst.loadNN(save_path);

   # >>>>> Test Neural Network <<<<<
   mlpnnocr_inst.testNN();

   # >>>>> Demonstration <<<<<
#   mlpnnocr_inst.demo(demoConfig);

   # >>>>> Program Finished! <<<<<
   print("Done!");

# /////////////////////////////////////////////////////////////////////////////////////////////////////
#                                      Application Entry Point
# /////////////////////////////////////////////////////////////////////////////////////////////////////
if __name__ == "__main__":

   main();
   



   

   
      
      















