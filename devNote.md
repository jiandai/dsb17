##Development Diary 
- 2017-1-13-fri set the repo ~ 12:32:17 PM
- ... ...
- 2017-2-15-wed present in DL grp meeting
- 2017-2-17-fri wrap up py2oracle
- 2017-2-18-sat py packing, OO paradigm
- ... ...
- 2017-2-19-sun cuda, and R-cuda interface
- 2017-2-20-mon Unlearn stat :) so funny, presidents day
- 2017-2-21-tue Dahshu conf, Thomas asked to focus on DSB17 data, image pipeline
- 2017-2-22-wed note DL grp meeting: imageJ/nifity/osirix, start to work on dicom metadata ref on pydicom: http://pydicom.readthedocs.io/en/stable/getting_started.html
- 2017-2-23-thu dicom metadata, patient level vs slice level
- 2017-2-24-fri dicom pixel data, dicom geometry
- 2017-2-25-sat conversion, resampling and normalization 
- ... ...
- 2017-2-26-sun check training and test sets
- 2017-2-27-mon separate training from test
- 2017-2-28-tue resolve the storage issue by using workspace on server, download data to server per wget + cookies file => review the time line
- 2017-3-1-wed extract the data by 7z, more data from kaggle, revamp MNIST test case on server
- 2017-3-2-thu start to use kernel feature of kaggle /w MNIST
- 2017-3-3-fri ML methodology review: CV, etc
- ... ...
- 2017-3-5-sun reading on 3d ct data processing
- 2017-3-6-mon work on the nonsense/idiotic/naive baseline, and tensorflow installation, use new server to run tensorflow
- 2017-3-7-tue reading preprocess and segmentation, kernel test
- 2017-3-8-wed DL meeting, Priya joined, theano broken on `new server'
- 2017-3-9-thu LSF array, install pydicom on server



## Ref

### DL libs:
- caffe: 
http://christopher5106.github.io/deep/learning/2015/09/04/Deep-learning-tutorial-on-Caffe-Technology.html
http://christopher5106.github.io/big/data/2015/07/16/deep-learning-install-caffe-cudnn-cuda-for-digits-python-on-ubuntu-14-04.html
- keras
- tensorflow
- theano

- cntx
- mxnet

- torch
- Deeplearning4j

### Convolutional layer
Phi^I_ab = \sum_x \sum_y W^{IJ}_{ab,xy} phi^J_{xy}+ b^I
