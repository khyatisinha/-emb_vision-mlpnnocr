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
nTrainSamples  = 10000;
maxCharLength  = 4;
maxWordLength  = 3;
data_directory = '../data';
nnShape        = (10,10);  # good (80,36)
save_path      = '../model/mlpnnocr_model.pkl';


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
   mlpnnocr_inst = MlpnnOcr(data_directory);

   # >>>>> Generate Training/Testing Data If It Does Not Exist <<<<<
   if not os.path.exists(data_directory):
      mlpnnocr_inst.genTrainingTestingData(nTrainSamples, maxCharLength, maxWordLength);

   # >>>>> Do stuff with it <<<<<
   mlpnnocr_inst.projectInfo();

   # >>>>> Import Training/Testing Data <<<<<
#   mlpnnocr_inst.importTrainingTestingData();
#
#   # >>>>> Build training/testing arrays <<<<<
#   mlpnnocr_inst.buildTrainTestArrays();

#   # >>>>> Create Neural Network <<<<<
#   mlpnnocr_inst.createNN(nnShape);
#   
#   # >>>>> Train Neural Network <<<<<
#   mlpnnocr_inst.trainNN(save_path);

   # >>>>> Create Neural Network <<<<<
   mlpnnocr_inst.loadNN(save_path);
   
   # >>>>> Test Neural Network <<<<<
#   mlpnnocr_inst.testNN();
   
   # >>>>> Classify Something <<<<<
   randExample = mlpnnocr_inst.genRandomTextImage(maxCharLength, maxWordLength);
   prediction  = mlpnnocr_inst.classifyNN(randExample['image'], randExample['words']);

   # >>>>> Program Finished! <<<<<
   print("Done!");

# /////////////////////////////////////////////////////////////////////////////////////////////////////
#                                      Application Entry Point
# /////////////////////////////////////////////////////////////////////////////////////////////////////
if __name__ == "__main__":

   main();
   



   

   
      
      















