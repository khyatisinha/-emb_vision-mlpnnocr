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
#  Description:  Creates an instance of the MlpnnOcr class, loads a saved MLPNN sklearn neural network
#                and demonstrates the real-world video frame experiment.
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

# Possible characters
load_path          = '../model/mlpnnocr_model.pkl';  # File to save trained neural network results to
alpha_chars        = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','&','@'];

# >>>>> Demonstration Configuration <<<<<
demoConfig = {
              'n_demo_samples'    : 10000        ,    # Only needed for Windows, to prevent sklearn fortran from crashing command prompt.
              'demo_period'       : 0.001        ,    # The amount of time between examples during demonstration
              'camera_pos'        : 0            ,    # Camera operating system position to use, if -1, no camera
              'resolution'        : (640,480)    ,    # Camera resolution, if used in demo
              'max_char'          : 4            ,    # Maximum number of characters in a word
              'max_words'         : 3            ,    # Maximum number of words in demo examples
              'low_thresh'        : 120          ,    # Low threshold value for cv2.threshold();
              'high_thresh'       : 255          ,    # High threshold value for cv2.threshold();
              'make_video'        : False        ,    # True->record experiment, False->do not record experiment
              'video_resolution'  : (640,820)    ,    # Camera resolution, if used in demo
              'save_path'         : 'demo_video.avi', # Path including filename to save experiment video to
              'fps'               : 5.0               # Frames per second of experiment video
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
   mlpnnocr_inst = MlpnnOcr(alpha_chars);

   # >>>>> Project Information <<<<<
   mlpnnocr_inst.projectInfo();

   # >>>>> Create Neural Network <<<<<
   mlpnnocr_inst.loadNN(load_path);
   
   # >>>>> Demonstration <<<<<
   mlpnnocr_inst.demo(demoConfig);

   # >>>>> Program Finished! <<<<<
   print("Done!");

# /////////////////////////////////////////////////////////////////////////////////////////////////////
#                                      Application Entry Point
# /////////////////////////////////////////////////////////////////////////////////////////////////////
if __name__ == "__main__":

   main();
   



   

   
      
      















