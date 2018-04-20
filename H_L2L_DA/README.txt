In this directory there are the matlab scripts we used for the 
experiments described in CVPR 2014: 
"Learning to Learn, from Transfer Learning to Domain Adaptation: A Unifying Perspective".
Cite this paper as reference for the software.

This code has been tested on Matlab 2013a on Linux.

The code is part of H-L2L for Domain Adaptation, refer to section 5.1 in the paper. 

Directory structure
- classifier: script of classifier
- data: amazon, caltech, dslr, webcam with 10 classes
- script: H-L2L algorithm

To run the code
1. run generate_splits.m to create split files for all data
2. run find_bestC for finding best C, example: find_bestC('amazon')
3. in demo.m, change sources and target name
4. run demo.m

For running Office dataset with 31 classes, please refer to 
http://www1.icsi.berkeley.edu/~saenko/projects.html#data







