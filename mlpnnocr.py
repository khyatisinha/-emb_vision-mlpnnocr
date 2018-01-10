#!/usr/bin/env python
'''
# /////////////////////////////////////////////////////////////////////////////////////////////////////
#                               MLPNN Optical Character Recognition Class
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
#  Description:  A class for creating an optical character recognition system. Can generate data
#                for training/testing an MLPNN, create the MLPNN, generate random images, and more.
#                Two demonstration mode are supported; 1.) random text images and 2.) real-world video
#                frame classification.
#
#     SVN Info:
#                $Author: jsackos $
#                $Date: 2017-12-07 19:33:31 -0800 (Thu, 07 Dec 2017) $
#                $Revision: 90 $
#                $Id: mlpnnocr.py 90 2017-12-08 03:33:31Z jsackos $
#                $Header: https://ksinha@projects.cecs.pdx.edu/svn/jsackos-mlpnnocr/python/mlpnnocr.py 90 2017-12-08 03:33:31Z jsackos $
#                $HeadURL: https://ksinha@projects.cecs.pdx.edu/svn/jsackos-mlpnnocr/python/mlpnnocr.py $
#
# /////////////////////////////////////////////////////////////////////////////////////////////////////
'''

# /////////////////////////////////////////////////////////////////////////////////////////////////////
#                                               Imports
# /////////////////////////////////////////////////////////////////////////////////////////////////////
import sys, os, time, random, datetime
from collections import OrderedDict
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
import pickle
import cv2

# /////////////////////////////////////////////////////////////////////////////////////////////////////
#                               MLPNN Optical Character Recognition Class
# /////////////////////////////////////////////////////////////////////////////////////////////////////
class MlpnnOcr():

   '''
   # //////////////////////////////////////////////////////////////////////////////////////////////////
   #                                        Class Data Members
   # //////////////////////////////////////////////////////////////////////////////////////////////////
   '''
   character_pixel_padding   = 50;      # Number of empty pixels on left, right, top, bottom of generated image
   char_pixel_width          = 80;      # Assumed maximum pixel width of a HERSHEY font character
   char_pixel_height         = 80;      # Assumed maximum pixel height of a HERSHEY font character
   low_thresh                = 150;     # Lower bound for pixel intensity, i.e. set everything below it to black.
   high_thresh               = 255;     # Need to rename, is actually the new value if above threshold
   txt_x_offset              = 50;      # Distance from the left side of the image to begin inserting text
   minArea                   = 175;     # Lower bound for char area threshold, noise should be smaller than this
   maxArea                   = 1500;    # Upper bound for char area threshold, large objects get filtered out
   rowCmpPercent             = 0.6;     # Percentage of char sub-image height to use for determing row elements
   charSpaceDetMult          = 1.9      # Char width multiplier to use for determining if space present

   '''
   # //////////////////////////////////////////////////////////////////////////////////////////////////
   #                                         Class Constructor
   #
   #   Description:  This task is what gets called when you instance a class.  All work for creating
   #                 a new instance of the class should go here.
   #     
   #     Arguments:  N/A
   #       
   #       Returns:  N/A
   #          
   # //////////////////////////////////////////////////////////////////////////////////////////////////
   '''
   def __init__(self, alpha_chars, data_directory='../data'):

      # >>>>> Assign Input Arguments <<<<<
      self.data_dir = data_directory;

      # >>>>> Seed Random Number Generator <<<<<
      np.random.seed(int(time.time()));

      # >>>>> Array of Possible Characters <<<<<
      self.alpha_chars = alpha_chars;
      
      # >>>>> Create Data Members <<<<<
      self.train_data              = [];      # Training data array
      self.test_data               = [];      # Testing data array
      self.train_vectors           = [[],[]]; # Array of training vector subimage/target
                                              # self.train_vectors[0] is numpy array of subimages
                                              # self.train_vectors[1] is numpy array of targets, i.e. answers
      self.test_vectors            = [[],[]]; # Array of testing vector subimage/target
                                              # self.test_vectors[0] is numpy array of subimages
                                              # self.test_vectors[1] is numpy array of targets, i.e. answers
      self.mlpnn                   = False;
      self.test_conf_matrix        = False;
      self.realtime_confusion_mat  = np.zeros((len(self.alpha_chars),len(self.alpha_chars)));
      self.video                   = False;
      self.video_writer            = False;
      self.display_window          = False;
      self.window_name             = 'MLPNN Optical Character Recognition Demo';
      
      # >>>>> Font Configuration <<<<<
      self.fontFace      = cv2.FONT_HERSHEY_SIMPLEX;
      self.fontScale     = 2;
      self.fontThickness = 2;
      
      # >>>>> Performance Data Members <<<<<
      self.ave_mlpnn_predict_samples = 0;
      self.ave_mlpnn_predict_time    = 0.0;
      self.ave_classifyNN_samples    = 0;
      self.ave_classifyNN_time       = 0.0;
      self.ave_display_samples       = 0;
      self.ave_display_time          = 0.0;
   '''
   # //////////////////////////////////////////////////////////////////////////////////////////////////
   #                                         Class Destructor
   #
   #   Description:  Perform any tasks that should be completed before a class instance is destroyed.
   #
   #     Arguments:  N/A
   #
   #       Returns:  N/A
   #
   # //////////////////////////////////////////////////////////////////////////////////////////////////
   '''
   def __del__(self):
      pass;

   '''
   # //////////////////////////////////////////////////////////////////////////////////////////////////
   #                                           printHeader()
   #
   #   Description:  Highlight a section in text display, by printing a header with horizontal
   #                 character dividers.
   #
   #     Arguments:  msg  : string : The message to be printed.
   #                 char : string : The character symbol to use for the header.
   #
   #       Returns:  N/A
   #
   # //////////////////////////////////////////////////////////////////////////////////////////////////
   '''
   def printHeader(self, msg, char='*'):
      msg_len       = len(msg);
      horiz_divider = char*(msg_len+10*2);
      spaces        = ' '*10;
      print('\n' + horiz_divider);
      print(spaces + msg);
      print(horiz_divider);
      
      
   '''
   # //////////////////////////////////////////////////////////////////////////////////////////////////
   #                                           projectInfo()
   #
   #   Description:  A method for displaying the authors and description of the project.
   #
   #     Arguments:  N/A
   #
   #       Returns:  N/A
   #
   # //////////////////////////////////////////////////////////////////////////////////////////////////
   '''
   def projectInfo(self):
      self.printHeader('MLPNN Optical Character Recognition');
      print("    Authors:  Josh Sackos, Khyati Sinha");
      print("     Course:  Embedded Vision 1, ECE510-002");
      print("    Project:  MLPNN Optical Character Recognition");
      print("Description:  An optical character recognition application for detecting/classifying digits 0-9, uppercase letters A-Z, and symbols '@' and '&'.\n");

   '''
   # //////////////////////////////////////////////////////////////////////////////////////////////////
   #                                         genRandomText()
   #
   #   Description:  Generates a random string from the self.alpha_chars data member, of a variable
   #                 length, and then returns the string.
   #
   #     Arguments:  
   #                 nChars : integer : Maximum number of random characters to generate per word.
   #                 nWords : integer : Maximum number of words to generate.
   #
   #       Returns:  string : Random text that was generated.
   #
   # //////////////////////////////////////////////////////////////////////////////////////////////////
   '''
   def genRandomText(self, nChars=4, nWords=1):
   
      # Generate the text for the image
      text        = '';
      # Generate random number of words, up to nWords
      nWords      = np.random.randint(1,nWords+1);
      for i in range(0, nWords):
         # Random word length
         tmpLength = np.random.randint(1, nChars+1);

         # Generate random word
         rndIdx = np.random.randint(0, len(self.alpha_chars), tmpLength);
         for idx in rndIdx:
            text += self.alpha_chars[idx];
         
         # Add space if not last word
         if i != nWords - 1:
            text += ' ';

      return text;
      
   '''
   # //////////////////////////////////////////////////////////////////////////////////////////////////
   #                                          dispImage()
   #
   #   Description:  Displays the image passed in, for a specified amount of time.
   #
   #     Arguments:  
   #                 img           : numpy array : The OpenCV/numpy array that contains image data.
   #                 title         : str         : Title of the window that will display the image.
   #                 display_delay : float       : The amount of seconds to delay by before returning.
   #
   #       Returns:  N/A
   #
   # //////////////////////////////////////////////////////////////////////////////////////////////////
   '''
   def dispImage(self, img, title, display_delay=0):

      if not self.display_window:
         cv2.namedWindow(self.window_name,cv2.WINDOW_NORMAL); # Create an OpenCV window
   
      display_delay = int(display_delay*1000);
      if not isinstance(img, list):
         img = [img];
      i = 0;
      for image in img:
         cv2.resizeWindow(self.window_name, image.shape[1],image.shape[0]);
         cv2.imshow(self.window_name, image);                   # Show the image in the window
         i += 1;
      if self.video_writer:
         print("write frame");
         print(type(img[0]));
         self.video_writer.write(img[0]);
      cv2.waitKey(display_delay);               # Wait for user to press a button
#      cv2.destroyAllWindows();                  # Destroy the window
      
   '''
   # //////////////////////////////////////////////////////////////////////////////////////////////////
   #                                         genTextImage()
   #
   #   Description:  Generates an image with reslution "width" x "height", with "text"
   #                 superimposed in it.
   #
   #     Arguments:  
   #                 text   : string      : Text to superimpose into an image.
   #                 width  : integer     : Pixel width of the image window.
   #                 height : integer     : Pixel height of the image window.
   #                 length : integer     : Number of characters per word.
   #                 nWords : integer     : Number of words in the image.
   #                 color  : tuple(,,)   : BGR color of text.
   #                 bgcolor: tuple(,,)   : BGR background color of image.
   #
   #       Returns:  N/A
   #
   # //////////////////////////////////////////////////////////////////////////////////////////////////
   '''
   def genTextImage(self, text, width, height, length=4, nWords=1, color=(0,0,0), bgcolor=255):
      width      += 2*self.character_pixel_padding; # Left and right padding
      height     += 2*self.character_pixel_padding; # Top and bottom padding
      
      img = np.full((height, width, 3), bgcolor, np.uint8);    # 3-channel white image of resolution width x height
      txt_origin  = (self.txt_x_offset, int(height/2.0)+int(self.char_pixel_height*0.333));

      cv2.putText(img,text,txt_origin, self.fontFace, self.fontScale, color, 2, cv2.LINE_AA);
      
      return img;

   '''
   # //////////////////////////////////////////////////////////////////////////////////////////////////
   #                                      genRandomTextImage()
   #
   #   Description:  Generates an image with random text superimposed in it.  Use for development,
   #                 and live demonstration.
   #
   #     Arguments:  
   #                 maxLength : integer : Maximum number of characters per word.
   #                 maxWords  : integer : Maximum number of words in the image.
   #
   #       Returns:  OrderedDict : Ordered dictionary containing the generated image and the ASCII
   #                               targets.
   #
   # //////////////////////////////////////////////////////////////////////////////////////////////////
   '''
   def genRandomTextImage(self, maxLength=4, maxWords=1):

      # >>>>> Make sure arguments are valid <<<<<
      if maxLength <= 0:
         print("Illegal maxLength = %d, must be greater than or equal to 1" % maxLength);
         return;
      elif maxWords    <= 0:
         print("Illegal maxWords = %d, must be greater than or equal to 1" % maxWords);
         return;

      maxSpaces = maxWords-1;                                   # The maximum number of spaces that may be present
      maxChars  = maxLength*maxWords;                           # The maximum number of characters that may be present
      width     = (maxSpaces + maxChars)*self.char_pixel_width; # The maximum number of spaces and characters pixel width
      height    = self.char_pixel_height;
      color     = (0,0,0);
      bgcolor   = 255;

      # Generate the text for the image
      text = self.genRandomText(maxLength, maxWords);
      
      # Create an image with the text superimposed
      img = self.genTextImage(text,width,height,maxLength,maxWords,color,bgcolor);

      return OrderedDict([ ('image',img),('words',text.split(' ')) ]);
      
   '''
   # //////////////////////////////////////////////////////////////////////////////////////////////////
   #                                      genTrainingTestingData()
   #
   #   Description:  Generates parameterized number of stochastic trianing/testing data examples,
   #                 for training and testing the neural network. Each example is saved to disk
   #                 in the self.data_dir directory.
   #
   #     Arguments:  
   #                 nSamples  : integer : Number of training/testing exapmles to generate
   #                 maxLength : integer : Maximum number of characters per word.
   #                 maxWords  : integer : Maximum number of words in the image.
   #
   #       Returns:  N/A
   #
   # //////////////////////////////////////////////////////////////////////////////////////////////////
   '''
   def genTrainingTestingData(self, nSamples, maxLength=4, maxWords=1):

      self.printHeader("Generating %d Training/Testing Examples" % nSamples);
   
      # >>>>> Make sure arguments are valid <<<<<
      if nSamples    <= 0:
         print("Illegal nSamples = %d, must be great than or equal to 1" % nSamples);
      elif maxLength <= 0:
         print("Illegal maxLength = %d, must be greater than or equal to 1" % maxLength);
         return;
      elif maxWords  <= 0:
         print("Illegal maxWords = %d, must be greater than or equal to 1" % maxWords);
         return;

      maxSpaces         = maxWords-1;                          # The maximum number of spaces that may be present
      maxChars          = maxLength*maxWords;                  # The maximum number of characters that may be present
      color   = (0,0,0);
      bgcolor = 255;
      
      # Make sure data directory exists, if not create it
      if not os.path.exists(self.data_dir):
         os.makedirs(self.data_dir);
      
      # >>>>> Generate nSamples <<<<<
      for k in range(0, nSamples):

         # Generate the text for the image
         text = self.genRandomText(maxLength, maxWords);
         width, height = cv2.getTextSize(text, self.fontFace, self.fontScale, self.fontThickness)[0];
         width  = width + self.txt_x_offset;
         height = height;

         # Create an image with the text superimposed
         img = self.genTextImage(text,width,height,maxLength,maxWords,color,bgcolor);
         
         # >>>>> Skew the image randomly, may not be skewed! <<<<<
         img = self.randAffine(img);

         # Create file name that has answer in it.
         fName = 'img_' + str(k) + '_' + text.replace(' ', '_') + '.jpg';

         # Save image to disk
         cv2.imwrite(self.data_dir + '/' + fName, img);

   '''
   # //////////////////////////////////////////////////////////////////////////////////////////////////
   #                                          randAffine()
   #
   #   Description:  Applies a randomly generated affine matrix, and applies it to an input image.
   #                 The modified image is returned.
   #
   #     Arguments:  
   #                 img : ndarray : The image to be transformed.
   #
   #       Returns:  img : ndarray : Modified image with random affine transformation applied.
   #
   # //////////////////////////////////////////////////////////////////////////////////////////////////
   '''
   def randAffine(self, img):

      # Translate, Rotate, Skew, etc.
      transforms = ['none', 'affine'];
      rndTrans   = transforms[np.random.randint(0,len(transforms))];
      rows, cols = img.shape[:2];
      if rndTrans == 'affine':
         src_points                   = np.float32([[0,0], [cols-1,rows-1], [cols-1,0] ]);
         right_x                      = img.shape[1]*np.random.uniform(low=0, high=.65);
         right_y                      = img.shape[0]*np.random.uniform(0.01,0.3);
         left_y                       = np.random.uniform(low=0.0, high=0.4);

         dst_points                   = np.float32([[0,img.shape[0]*left_y], [cols-right_x,rows-right_y], [cols-right_x,right_y] ]);

         aff_mtx                      = cv2.getAffineTransform(src_points, dst_points);
         img                          = cv2.warpAffine(img, aff_mtx, (cols,rows));
#            cv2.circle(img, tuple(dst_points[0]), 10, (0,255,0), 2);
#            cv2.circle(img, tuple(dst_points[1]), 10, (0,255,0), 2);
#            cv2.circle(img, tuple(dst_points[2]), 10, (0,255,0), 2);

      # >>>>> Make sure corners are filled with white <<<<<
      # img, mask, seed point, newVal, loDiff, upDiff, rect, flags
      mask = np.full((img.shape[0]+2,img.shape[1]+2), 0, dtype=np.uint8);
      flags = 0;
      retval, img, mask, rect = cv2.floodFill(img, mask, (0,0)                           , (255,255,255), (0,0,0), (255,255,255), flags);
      retval, img, mask, rect = cv2.floodFill(img, mask, (0, img.shape[0]-1)             , (255,255,255), (0,0,0), (255,255,255), flags);
      retval, img, mask, rect = cv2.floodFill(img, mask, (img.shape[1]-1,0)              , (255,255,255), (0,0,0), (255,255,255), flags);
      retval, img, mask, rect = cv2.floodFill(img, mask, (img.shape[1]-1, img.shape[0]-1), (255,255,255), (0,0,0), (255,255,255), flags);
      
      return img;

   '''
   # //////////////////////////////////////////////////////////////////////////////////////////////////
   #                                  importTrainingTestingData()
   #
   #   Description:  Imports all JPEG images in self.data_dir, uses all filename text appearing after
   #                 "img_XXXX_" in filename to determine the correct answer for the example.  Creates
   #                 two arrays:
   #
   #                             train_data  : Array for storing images
   #                             test_data   : Array for storing the solutions/answers/targets
   #
   #     Arguments:  trainPercentage : float : Percentage of images to use for training, rest are
   #                                           used for testing.
   #
   #       Returns:  N/A
   #
   # //////////////////////////////////////////////////////////////////////////////////////////////////
   '''
   def importTrainingTestingData(self, trainPercentage=60.0):

      files           = [];
      self.train_data = [];
      self.test_data  = [];
      
      self.printHeader("Importing training/testing data from directory:  %s" % self.data_dir);
      sys.stdout.flush();
   
      # >>>>> Make sure directory exists, if not, alert user and abort <<<<<
      if not os.path.exists(self.data_dir):
         print("Error:  %s directory does not exist, aborting..." % self.data_dir);
         return;
      
      # >>>>> Get list of directory contents <<<<<
      for item in os.scandir(self.data_dir):
         # Skip all items that are not JPEG
         if item.is_dir() or item.name.find('.jpg') == -1:
            print("skipped non-jpeg item: %s" % item.name);
            continue;

         files.append(item);

      # >>>>> Shuffle Array <<<<<
      np.random.shuffle(files);

      # >>>>> Split Files Between Training/Testing <<<<<
      nTrain    = int(len(files)*trainPercentage/100.0);
      nTest     = len(files)-nTrain;
      trainList = files[0:nTrain];
      testList  = files[nTrain:];
      
      # >>>>> Import Training Set <<<<<
      self.dataDistr = OrderedDict();
      for item in trainList:
         tmpData   = cv2.imread(item.path, cv2.IMREAD_COLOR);
         tmpTarget = item.name.replace('.jpg', '').split('_')[2:];
         if not tmpTarget[0] in self.dataDistr.keys():
            self.dataDistr[tmpTarget[0]] = {'train':1, 'test':0};
         else:
            self.dataDistr[tmpTarget[0]]['train'] += 1;
         
         self.train_data.append(OrderedDict([ ('image',tmpData),('words',tmpTarget) ]));

      # >>>>> Import Testing Set <<<<<
      for item in testList:
         tmpData   = cv2.imread(item.path, cv2.IMREAD_COLOR);
         tmpTarget = item.name.replace('.jpg', '').split('_')[2:];
         self.test_data.append(OrderedDict([ ('image',tmpData),('words',tmpTarget) ]));
         if not tmpTarget[0] in self.dataDistr.keys():
            self.dataDistr[tmpTarget[0]] = {'train':0, 'test':1};
         else:
            self.dataDistr[tmpTarget[0]]['test'] += 1;

      # >>>>> Save Data Statistics to Disk <<<<<
      f = open('../model/last_train_test_data_split.txt', 'w');
      f.write("      Train    Test    Total");
      f.write("-------------------------------------------");
      for key, val in self.dataDistr.items():
         tmpStr = str(key)+" "+"%4s" % str(val['train'])+' '+"%4s" % str(val['test'])+' '+"%4s" % str(val['train']+val['test']) + '\n';
         f.write(tmpStr); 
      f.close();
      print("Import complete!");

      return;
      
   '''
   # //////////////////////////////////////////////////////////////////////////////////////////////////
   #                                       extractChars()
   #
   #   Description:  Tries to extract characters from the input image argument, in order, and returns
   #                 the characters as both text and sub-images, in an array.
   #
   #                             img_array  : Array for storing images
   #                             soln_array : Array for storing the solutions/answers/targets
   #
   #     Arguments:  
   #                 img : numpy array : Numpy array with the image to be processed.
   #
   #       Returns:  new_img   : ndarray     : Numpy array with contents of procesed image.
   #                 subimages : list        : List of ndarray sub-images for potential chracters.
   #                 rowDict   : OrderedDict : An ordered dictionary of the rows detected, with sorted
   #                                           sub-image lists in each row, contours and bounding rect.
   #
   # //////////////////////////////////////////////////////////////////////////////////////////////////
   '''
   def extractChars(self, img):

      # >>>>> Convert to Grayscale <<<<<
      gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
      
      # >>>>> Opening Operation <<<<<
      new_img = cv2.erode(gray_img, np.ones((3,3), np.uint8), iterations=1);
      new_img = cv2.dilate(new_img, np.ones((3,3), np.uint8), iterations=1);
      
      # >>>>> Threshold Operation <<<<<
      ret, new_img = cv2.threshold(gray_img, self.low_thresh, self.high_thresh, cv2.THRESH_BINARY_INV);
      
      # >>>>> Find Countours <<<<<
      new_img, contours, hierarchy = cv2.findContours(new_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE);

      # ////////////////////////////////////////////////
      #        Sort Contours left/right top/bottom
      # ////////////////////////////////////////////////

      # >>>>> Separate Rows <<<<<
      rowDict = OrderedDict();
      del_list = [];
      row_label = False;
      for i in range(0, len(contours)):
         x,y,w,h = cv2.boundingRect(contours[i]);
         if (w*h < self.minArea or w*h > self.maxArea) and self.video:
            del_list.append(i);
            i = i -1;
            continue;
            
         # ** Assign to a row ** 
         if row_label:
            if not (y >= row_label - h*self.rowCmpPercent) and (y <= row_label + h + h*self.rowCmpPercent):
               row_label = y;
         else:
            row_label = y;
         
         
         if not (row_label in rowDict.keys()):
            rowDict[row_label] = [];
         rowDict[row_label].append({'contour':contours[i], 'bound_rect':(x,y,w,h)});
      rowDict = OrderedDict(sorted(rowDict.items()));

      # >>>>> Sort Each Row <<<<<
      for row_label, row in rowDict.items():
         colDict = OrderedDict();
         # For each character in the row
         for contour in row:
            x = contour['bound_rect'][0];
            colDict[x] = contour;
         rowDict[row_label] = OrderedDict(sorted(colDict.items()));

      # ////////////////////////////////////////////////
      #                Extract Subimages
      # ////////////////////////////////////////////////
      subimages = [];
      for row_label, row in rowDict.items():
         for col_label, contour in row.items():
            x,y,w,h = contour['bound_rect'];
            cv2.rectangle(new_img,(x,y),(x+w,y+h),(128),1);
            subimage = gray_img[y:y+h, x:x+w];

            deltaW = int(self.char_pixel_width  - subimage.shape[1]);
            deltaH = int(self.char_pixel_height - subimage.shape[0]);
            lAdd   = int(deltaW/2.0);
            rAdd   = int(deltaW - lAdd);
            tAdd   = int(deltaH/2.0);
            bAdd   = int(deltaH-tAdd);

            if lAdd < 0 or rAdd < 0 or tAdd < 0 or bAdd < 0:
               self.printHeader("%f %f %f %f" % (lAdd,rAdd,tAdd,bAdd));
               
               
            if not (subimage.shape[1] > self.char_pixel_width) and not (subimage.shape[0] > self.char_pixel_height):
               proc_subimage = cv2.copyMakeBorder(subimage,tAdd,bAdd,lAdd,rAdd,cv2.BORDER_CONSTANT, value=[255]);

            median = (np.amax(proc_subimage) - np.amin(proc_subimage))/2.0;
            ret, proc_subimage = cv2.threshold(proc_subimage, median-median*0.05, 255, cv2.THRESH_BINARY);
         
   #         proc_subimage = cv2.adaptiveThreshold(proc_subimage,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,3,5);
            subimages.append(proc_subimage);

#      self.dispImage(subimages, 'Extracted Characters');

      return (new_img, subimages, rowDict);

   '''
   # //////////////////////////////////////////////////////////////////////////////////////////////////
   #                                     buildTrainTestArrays()
   #
   #   Description:  Processes all training/testing samples, by extracting the characters for each
   #                 each sample, and appending each subimage/target value to a list.  Training
   #                 samples get appended to the train_vectors list, and testing samples get appended
   #                 to the test_vectors list.  These lists are returned, even though they are stored
   #                 as internal data members.
   #
   #     Arguments:  N/A
   #
   #       Returns:  N/A
   #
   # //////////////////////////////////////////////////////////////////////////////////////////////////
   '''
   def buildTrainTestArrays(self):
   
      # >>>>> Clear old data <<<<<
      self.train_vectors = [[],[]]; # Array of training vector subimage/target
      self.test_vectors  = [[],[]]; # Array of testing vector subimage/target
      
      self.printHeader("Building Training/Testing Vectors");
      sys.stdout.flush();

      for train_sample in self.train_data:
         proc_img, subimages, rowDict = self.extractChars(train_sample['image']);
         char_list           = np.array(list(''.join(train_sample['words'])));

         for i in range(0, len(subimages)):
            self.train_vectors[0] += [subimages[i]];
            self.train_vectors[1] += [char_list[i]];

      self.train_vectors[0] = np.array(self.train_vectors[0]);
      self.train_vectors[1] = np.array(self.train_vectors[1]);

      print("Finished creating training vectors!");
      print("#Subimages = ", len(self.train_vectors[0]));
      print("#Targets   = ", len(self.train_vectors[1]));
      print("\n");
      
      print("Building testing vector arrays...");
      sys.stdout.flush();

      for test_sample in self.test_data:
         proc_img, subimages, rowDict = self.extractChars(test_sample['image']);
         char_list           = np.array(list(''.join(test_sample['words'])));
         for i in range(0, len(subimages)):
            self.test_vectors[0] += [subimages[i]];
            self.test_vectors[1] += [char_list[i]];
      self.test_vectors[0] = np.array(self.test_vectors[0]);
      self.test_vectors[1] = np.array(self.test_vectors[1]);

      print("Finished creating testing vectors!");
      print("#Subimages = ", len(self.test_vectors[0]));
      print("#Targets   = ", len(self.test_vectors[1]));

      print("Training/testing vector arrays complete!");
      print("\n");

   '''
   # //////////////////////////////////////////////////////////////////////////////////////////////////
   #                                       printConfusionMtx()
   #
   #   Description:  Neatly prints the contents of a Numpy sklearn confusion matrix array. Adapts
   #                 the width of each entry to the char length of the entry with the maximum value.
   #
   #     Arguments:  arr : ndarray : The NxN array to be printed.
   #
   #       Returns:  N/A
   #
   # //////////////////////////////////////////////////////////////////////////////////////////////////
   '''
   def printConfusionMtx(self, arr):
      max_len = len(str(np.max(arr)));
      fmtStr = '%{0}s'.format(max_len+1);
      for m in range(0, arr.shape[0]):
         tmpStr = '';
         for n in range(0, arr.shape[1]):
            tmpStr += fmtStr % str(arr[m][n]);
         print(tmpStr);
      
   '''
   # //////////////////////////////////////////////////////////////////////////////////////////////////
   #                                           normalize()
   #
   #   Description:  Converts a numpy array to float32, finds the maximum value in the array, and then
   #                 normalizes all data values to be between 0 and 1.
   #
   #     Arguments:  npArray : numpy array : Numpy array that needs to be normalized.
   #
   #       Returns:  npArray : numpy array : Normalized numpy array.
   #
   # //////////////////////////////////////////////////////////////////////////////////////////////////
   '''
   def normalize(self, npArray):

      # >>>>> Normalize Numpy Array <<<<<
      npArray = npArray.astype(np.float32);
      max     = np.amax(npArray);
      npArray = npArray/max;
      
      return npArray;
      
   '''
   # //////////////////////////////////////////////////////////////////////////////////////////////////
   #                                           confusionAccuracy()
   #
   #   Description:  This method computes the accuracy of an NxN matrix by summing the diagonal of the
   #                 matrix, dividing by the total number of entries in the matrix, and then multiplying
   #                 the result by 100%.
   #
   #     Arguments:  matrix   : ndarray : The confusion matrix to compute the accuracy for.
   #                 nSamples : int     : The total number of samples that were present, i.e. sum
   #                                      of all elements in the input array.
   #
   #       Returns:  accuracy : float   : The computed accuracy for  the confusion matrix.
   #
   # //////////////////////////////////////////////////////////////////////////////////////////////////
   '''
   def confusionAccuracy(self, matrix, nSamples):
      # Correct answers are the diagonal of the NxN matrix argument
      diag_sum = 0;
      for i in range(0, matrix.shape[1]):
         diag_sum += matrix[i][i];

      # Compute the accuracy relative to the number of test samples
      nSamples = int(np.sum(matrix));
      accuracy = diag_sum/nSamples*100;
      
      return accuracy;
      
   '''
   # //////////////////////////////////////////////////////////////////////////////////////////////////
   #                                           createNN()
   #
   #   Description:  Creates an neural network from the specified configuraiton dictionary passed in.
   #                 This method does not train the neural network, instead it merely creates the
   #                 approiate sklearn data structure to model an MLPNN.
   #
   #     Arguments:  nnConfig : dict : Dictionary of sklearn neural network parameters to be used.
   #
   #       Returns:  N/A
   #
   # //////////////////////////////////////////////////////////////////////////////////////////////////
   '''
   def createNN(self, nnConfig):
   
      # >>>>> Create an sklearn neural network <<<<<
      self.printHeader("Creating Neural Network");
      self.mlpnn = MLPClassifier(solver             = nnConfig['solver'],
                                 activation         = nnConfig['activation'],
                                 hidden_layer_sizes = nnConfig['hidden_layer_sizes'],
                                 learning_rate      = nnConfig['learning_rate'],
                                 learning_rate_init = nnConfig['learning_rate_init'],
                                 max_iter           = nnConfig['max_iter'],
                                 momentum           = nnConfig['momentum'],
                                 shuffle            = nnConfig['shuffle']
                                 
      );
      print(self.mlpnn);
      print("Finished creating neural network!");

   '''
   # //////////////////////////////////////////////////////////////////////////////////////////////////
   #                                             loadNN()
   #
   #   Description:  Loads the specified neural network from disk into the class instance.
   #
   #     Arguments:  fpath  : string   : File path of pickle ".pkl" file to load.
   #
   #       Returns:  N/A
   #
   # //////////////////////////////////////////////////////////////////////////////////////////////////
   '''
   def loadNN(self, fpath):

      # >>>>> Load an sklearn neural network <<<<<
      self.printHeader("Loading neural network %s..." % fpath);
      f          = open(fpath, "rb");
      self.mlpnn = pickle.loads(f.read());
      f.close();
      self.printHeader("MLPNN", "*");
      print(self.mlpnn);
      print("Finished loading neural network!");

   '''
   # //////////////////////////////////////////////////////////////////////////////////////////////////
   #                                            trainNN()
   #
   #   Description:  Trains the neural network specified by self.mlpnn, using the training data in
   #                 self.train_vectors, and saves the result to disk.
   #
   #     Arguments:  save_path : string : Path including filename to save the trained neural network
   #                                      model to.
   #
   #       Returns:  N/A
   #
   # //////////////////////////////////////////////////////////////////////////////////////////////////
   '''
   def trainNN(self, save_path='../model/mlpnnocr_model.pkl'):
   
      # >>>>> Separate file name from path <<<<<
      fname = save_path.replace('\\', '/').split('/')[-1];
      path  = '/'.join(save_path.replace('\\', '/').split('/')[:-1]);

      self.printHeader("Neural Network Training");

      start_time = datetime.datetime.now();
  
      # >>>>> Normalize Training Data <<<<<
      self.train_vectors[0] = self.normalize(self.train_vectors[0]);

      # >>>>> Flatten subimages <<<<<
      train_subimages = [];
      for vec in self.train_vectors[0]:
         train_subimages.append(vec.flatten());
         
      # >>>>> Create vector of self.alpha_chars indexes <<<<<
      train_char_idxs = [];
      for char in self.train_vectors[1]:
         train_char_idxs.append(self.alpha_chars.index(char));
      
      # >>>>> Use self.train_vectors and self.test_vectors <<<<<
      # NOTE: self.train_vectors[0] is numpy array of subimages
      #       self.train_vectors[1] is numpy array of targets, i.e. answers
      self.mlpnn.fit(train_subimages, train_char_idxs);
      
      # >>>>> Save the model to disk <<<<<
      if not os.path.exists(path):
         os.makedirs(os.path.abspath(path));
      f = open(path + '/' + fname, 'wb');
      f.write(pickle.dumps(self.mlpnn));
      f.close();
      
      print("Training Time = ", (datetime.datetime.now() - start_time).total_seconds(), " seconds.");
      
      print("Finished training neural network!");
      
   '''
   # //////////////////////////////////////////////////////////////////////////////////////////////////
   #                                            testNN()
   #
   #   Description:  Tests the neural network self.mlpnn that is currently loaded/set, with the
   #                 testing data in self.test_vectors.
   #
   #     Arguments:  N/A
   #
   #       Returns:  N/A
   #
   # //////////////////////////////////////////////////////////////////////////////////////////////////
   '''
   def testNN(self):

      self.printHeader("Neural Network Testing");

      start_time = datetime.datetime.now();
      
      # >>>>> Normalize Training Data <<<<<
      self.test_vectors[0] = self.normalize(self.test_vectors[0]);

      # >>>>> Flatten subimages <<<<<
      test_subimages = [];
      for vec in self.test_vectors[0]:
         test_subimages.append(vec.flatten());
         
      # >>>>> Create vector of self.alpha_chars indexes <<<<<
      test_char_idxs = [];
      for char in self.test_vectors[1]:
         test_char_idxs.append(self.alpha_chars.index(char));
      
      # >>>>> Use self.test_vectors <<<<<
      # NOTE: self.test_vectors[0] is numpy array of subimages
      #       self.test_vectors[1] is numpy array of targets, i.e. answers
      predictions = self.mlpnn.predict(test_subimages);
      
      self.test_conf_matrix = confusion_matrix(test_char_idxs,predictions, labels=[i for i in range(0, len(self.alpha_chars))]);
      
      # >>>>> Print Testing Statistics <<<<<
      self.printHeader('Confusion Matrix', '/');
      self.printConfusionMtx(self.test_conf_matrix);
      self.printHeader('Testing Classification Report', '/');
      print(classification_report(test_char_idxs, predictions, labels=[i for i in range(0, len(self.alpha_chars))], target_names=self.alpha_chars));
      self.printHeader('Testing Results', '/');
      print("Accuracy = %f" % self.confusionAccuracy(self.test_conf_matrix, len(test_char_idxs) ));
      
      
      print("Time     = ", (datetime.datetime.now() - start_time).total_seconds(), " seconds.");

   '''
   # //////////////////////////////////////////////////////////////////////////////////////////////////
   #                                          classifyNN()
   #
   #   Description:  Classifies an input image as one of the self.alpha_chars characters, using the
   #                 trained self.mlpnn object.
   #
   #     Arguments:  img        : ndarray : Numpy array of image containing text to be classified/extracted.
   #                 target     : list    : List of target ASCII characters, if known.
   #                 disp_delay : float   : Number of seconds to show the results image for.
   #
   #       Returns:  string : String of character predictions.
   #
   # //////////////////////////////////////////////////////////////////////////////////////////////////
   '''
   def classifyNN(self, img, target=False, disp_delay=0):

      if not self.video:
         self.printHeader("Classifying", '/');
      start_time = datetime.datetime.now();
      # >>>>> Extract chars from image <<<<<
      proc_img, char_subimages, rowDict = self.extractChars(img);
      if target:
         if isinstance(target, list):
            char_list = np.array(list(''.join(target)));
         else:
            char_list = np.array(list(target));
      char_subimages = np.array(char_subimages);   # Convert from list to numpy array
         
      # >>>>> Normalize char subimages <<<<<
      if len(char_subimages) > 0:
         char_subimages = self.normalize(char_subimages);
      else:
         return [];

      # >>>>> Flatten subimages <<<<<
      subimages = [];
      for vec in char_subimages:
         subimages.append(vec.flatten());
   
      # >>>>> Classify via self.mlpnn.predict() <<<<<
      mlpnn_predict_time = datetime.datetime.now();
      predictions = self.mlpnn.predict(subimages);
      self.ave_mlpnn_predict_samples += 1;
      mlpnn_time_delta                = datetime.datetime.now() - mlpnn_predict_time;
      self.ave_mlpnn_predict_time     = (self.ave_mlpnn_predict_time + (datetime.datetime.now() - mlpnn_predict_time).total_seconds())/self.ave_mlpnn_predict_samples;
      
      pred_str    = '';
      
      # >>>>> Create string representation of predictions <<<<<
      cRow      = 0;
      pred_str  = '';
      charCount = 0;
      
      for row_label, row in rowDict.items():
         cCol = 0;
         row_col_labels = list(row.keys());

         for col_label, char in row.items():
            # ** Compute distance between last char and current char, insert space as needed **
            if cCol != 0:
               pX, pY, pW, pH = row[row_col_labels[cCol-1]]['bound_rect'];   # Previous character bounding rect
               cX, cY, cW, cH = char['bound_rect'];                          # Previous character bounding rect
               deltaX = cX - pX;

               if deltaX > cW*self.charSpaceDetMult:
                  pred_str += ' ';
            else:
               if self.video:
                  pred_str += '  *';

            pred_str += self.alpha_chars[predictions[charCount]];
            cCol += 1;
            charCount += 1;
         cRow += 1;

      if not self.video:
         print("Predicted           = ", pred_str);
      
      # >>>>> Expand Processed Image for Displaying Predictions <<<<<
      proc_img = cv2.copyMakeBorder(proc_img,100,0,0,0,cv2.BORDER_CONSTANT, value=[0]);

      # >>>>> Superimpose Predictions Text in Upper Right of Processed Image <<<<<
      self.insertPredictions(proc_img, pred_str);

      disp_img = np.concatenate((cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), proc_img), axis=0);

      stop_time = (datetime.datetime.now() - start_time - mlpnn_time_delta).total_seconds();
      print("Classification Time = ", stop_time, " seconds.");
      
      self.ave_classifyNN_samples += 1;
      self.ave_classifyNN_time     = (self.ave_classifyNN_time+stop_time)/self.ave_classifyNN_samples;
      
      if target and not self.video:
         print("Target              = ",  ''.join(target));
         
      # >>>>> Display Results <<<<<
      self.ave_display_samples += 1;
      display_time = datetime.datetime.now();
      self.dispImage(disp_img, 'White = Original Image , Black = Processed Image',display_delay=disp_delay);
      self.ave_display_time     = (self.ave_display_time + (datetime.datetime.now() - display_time).total_seconds())/self.ave_display_samples;

      return predictions;

   '''
   # //////////////////////////////////////////////////////////////////////////////////////////////////
   #                                        insertPredictions()
   #
   #   Description:  Inserts text into an image, in the upper right of the picture.
   #
   #     Arguments:  img    : ndarray : Numpy array that holds image.
   #                 text   : string  : Text to superimpose into an image.
   #                 color  : int     : 8-bit color intensity.
   #                 size   : float   : Font size/scale of text predictions in image.
   #
   #       Returns:  ndarray : Image with predictions superimposed.
   #
   # //////////////////////////////////////////////////////////////////////////////////////////////////
   '''
   def insertPredictions(self, img, text, color=128, size=1):
      if len(text) > 1:
         text = 'Predictions: ' + text;
      else:
         text = 'Prediction: ' + text;
         
      img_width      = img.shape[1];
      
      thickness = self.fontThickness;
      boxsize   = cv2.getTextSize(text, self.fontFace, size, self.fontThickness)[0];
      while(boxsize[0] > img_width and self.video):
         size = size/2;
         thickness = thickness-1;
         boxsize   = cv2.getTextSize(text, self.fontFace, size/2, self.fontThickness)[0];
      
      left_bottom_x  = img_width - boxsize[0] - self.character_pixel_padding-300 if self.video else img_width - boxsize[0] - self.character_pixel_padding;
      left_bottom_y  = self.character_pixel_padding + round(self.char_pixel_height/2.0);
      origin         = (left_bottom_x, left_bottom_y);
      
      cv2.putText(img,text,origin, cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness, cv2.LINE_AA);
      return img;
      
   '''
   # //////////////////////////////////////////////////////////////////////////////////////////////////
   #                                         createVideoCapture()
   #
   #   Description:  Creates an OpenCV VideoCapture object, and stores it as a class data member.
   #
   #     Arguments:  pos  : int   : Camera position number in operating system
   #                 res  : tuple : res[0] is camera pixel width, res[1] is camera pixel height
   #
   #       Returns:  bool : True->opened camera , False->failed to open camera
   #
   # //////////////////////////////////////////////////////////////////////////////////////////////////
   '''
   def createVideoCapture(self, pos=0, res=False):
      try:
         self.video = cv2.VideoCapture(pos);
         if res:
            self.video.set(cv2.CAP_PROP_FRAME_WIDTH ,res[0]);
            self.video.set(cv2.CAP_PROP_FRAME_HEIGHT,res[1]);
         print("Opened camera %d..." % pos);
         return True;
      except:
         print("ERROR: Could not open camera %d!" % pos);
         return False;
         
   '''
   # //////////////////////////////////////////////////////////////////////////////////////////////////
   #                                         createVideoWriter()
   #
   #   Description:  Create an instance of the OpenCV VideoWriter class, for saving demonstration
   #                 videos.
   #
   #     Arguments:  save_path  : string : Path including filename to save video to.
   #                 fps        : int    : Camera position number in operating system
   #                 resolution : tuple  : tuple[0]->width, tuple[1]->height of video.
   #
   #       Returns:  bool : True-> success, False->failure
   #
   # //////////////////////////////////////////////////////////////////////////////////////////////////
   '''
   def createVideoWriter(self, save_path, fps, resolution):
      try:
         self.video_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'MJPG'), fps, resolution);
         print("Created video writer...");
         return True;
      except:
         print("ERROR: Could not create video writer!");
         return False;

   '''
   # //////////////////////////////////////////////////////////////////////////////////////////////////
   #                                               demo()
   #
   #   Description:  Executes either the random text demonstration or the real-world video frame
   #                 demonstration.  The value of the 'camera_pos' dictionary key in the input
   #                 argument determines which demo gets executed. A -1 runs the random text
   #                 demo, and a positive number indicates a camera position to use.
   #
   #     Arguments:  demoConfig : dict : Dictionary of configuration parameters for demonstration.
   #
   #       Returns:  N/A
   #
   # //////////////////////////////////////////////////////////////////////////////////////////////////
   '''
   def demo(self, demoConfig):
      
      self.printHeader("Demonstration Running");
      
      self.low_thresh  = demoConfig['low_thresh'];
      self.high_thresh = demoConfig['high_thresh'];

      if demoConfig['camera_pos'] >= 0:
         self.createVideoCapture(demoConfig['camera_pos'], demoConfig['resolution']);
      if demoConfig['make_video']:
         self.createVideoWriter(demoConfig['save_path'], demoConfig['fps'], demoConfig['video_resolution']);
      
      while(1):
#      for i in range(0,demoConfig['n_demo_samples']):
         try:
            img     = False;
            target  = False;

            # >>>>> Use Camera Image <<<<<
            if self.video:
               # There seems to be a buffer of 3 images for the C920, read three images to get a fresh one
               for j in range(0, 3):
                  ret, img = self.video.read();

            # >>>>> Generate an Image <<<<<
            else:
               randomExample = self.genRandomTextImage(demoConfig['max_char'], demoConfig['max_words']);
               img    = self.randAffine(randomExample['image']);
#               self.dispImage(img, 'affine');
               target = randomExample['words'];
               
            # >>>>> Classify the digits found in the image <<<<<
            prediction  = self.classifyNN(img, target, disp_delay=demoConfig['demo_period']);

         except KeyboardInterrupt:
            print("ctrl+c detected, terminating demonstration...");
            break;
         except:
            pass;

      print("ave_mlpnn_predict_time = %0.6f" % self.ave_mlpnn_predict_time);
      print("ave_classifyNN_time    = %0.6f" % self.ave_classifyNN_time);
      print("ave_display_time       = %0.6f" % self.ave_display_time);
            
      cv2.destroyAllWindows();                  # Destroy the window

         

      
# /////////////////////////////////////////////////////////////////////////////////////////////////////
#                                 Development Application Entry Point
# /////////////////////////////////////////////////////////////////////////////////////////////////////
if __name__ == "__main__":

   print("This area is inteded for testing methods as they are developed, not as a program main().\n");
   



   

   
      
      















