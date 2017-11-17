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
   mlpnnocr_inst = MlpnnOcr();
   
   # >>>>> Do stuff with it <<<<<
   mlpnnocr_inst.projectInfo();
   
   # >>>>> Generate 10 Random Images <<<<<
   i = 0;
   for i in range(0, 10):
      mlpnnocr_inst.genTextImage(1200,250,4,3);

   # >>>>> Program Finished! <<<<<
   print("Done!");

# /////////////////////////////////////////////////////////////////////////////////////////////////////
#                                      Application Entry Point
# /////////////////////////////////////////////////////////////////////////////////////////////////////
if __name__ == "__main__":

   main();
   



   

   
      
      















