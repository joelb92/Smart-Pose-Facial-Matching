README

"Facial Frontalization and Smart Matching Via Pose" by Joel Brogan and Vitomir Struc

Solution for QUIS-CAMPI dataset ICB-RW 2016 Challenge

MAIN DEPENDENCIES:
Matlab
Python 2.7
skimage
sklearn
OpenCV
dlib (with Python2.7 bindings)

MAIN FUNCTIONS:

***To build a frontalized database for the given dataset (required to extract features):
run_ICBRW_experiment.m (matlab)
	INPUTS:
	probePath
	probeAnnotationPath
	WatchlistPath
	WatchlistAnnotationPath
*NOTE: This process takes some time.  I have included mat files with pre-computed face data to speed up the process. This data can be recomputed if necessary.
		If you wish to use the included pre-computed files, simply point run_ICBRW_experiment probe and watchlist directories to the empty (non cropped) directories located in the 'data/' folder
*NOTE2: if this function begins throwing python errors, please restart Matlab with using a call like this:
DYLD_INSERT_LIBRARIES=<path to libtiff.5.dylib>:<path to libz.1.2.8.dylib>:<path to libSystem.B.dylib>:<path to libgfortran.3.dylib> matlab
This will fix dlib dependency issues when matlab calls python (only do this if you are thrown warnings)
*NOTE3: There was an annotation error in the watchlist annotations file.  It has been corrected in the version of the annotation file included.

***To generate features via the slmsimple Neural Network and build a similarity matrix:
slmsimple.py <image_watchlist_dir> <image_probe_dir> (python)

this file will generate a similarity matrix in a .csv format named SimilarityMatrix.csv


References:
[1]Cox, David, and Nicolas Pinto. "Beyond simple features: A large-scale feature search approach to unconstrained face recognition." Automatic Face & Gesture Recognition and Workshops (FG 2011), 2011 IEEE International Conference on. IEEE, 2011.
[2]Hassner, Tal, et al. "Effective face frontalization in unconstrained images." arXiv preprint arXiv:1411.7964 (2014).
[3]Kazemi, Vahdat, and Josephine Sullivan. "One millisecond face alignment with an ensemble of regression trees." Computer Vision and Pattern Recognition (CVPR), 2014 IEEE Conference on. IEEE, 2014.
[4]Zhu, Xiangxin, and Deva Ramanan. "Face detection, pose estimation, and landmark localization in the wild." Computer Vision and Pattern Recognition (CVPR), 2012 IEEE Conference on. IEEE, 2012.
