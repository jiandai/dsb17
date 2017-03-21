
##Summary

- preprocessing bottleneck (time consuming) : solved by LSF array (simple, 3/14)
- in-memory bottleneck (OOM) : solved by refactoring the code / optimizing the loop (simple, 3/17)
- cnn size bottleneck (size and training time) :
- algorithm bottleneck : SME input (?) by using segmentation

##Development Diary 

- ... ... 
- 2017-1-13-fri set the repo ~ 12:32:17 PM
- 2017-1-20-fri company ML meeting...
- 2017-1-26-thu kick-off DL meeting /w Liz & Thomas, run sample code for cifar10 classification
- ... ... 
- [prep DL libs]
- ... ... 
- 2017-2-14-tue start to run on gpu
- 2017-2-15-wed present in 1st DL grp meeting, some quick benchmark on gpu vs cpu
- 2017-2-17-fri wrap up py2oracle
- 2017-2-18-sat py packing, OO paradigm
- ... ... w0
- 2017-2-19-sun cuda, and R-cuda interface
- 2017-2-20-mon Unlearn stat :) so funny, presidents day, joint meeting /w CIG 
- 2017-2-21-tue Dahshu conf, Thomas asked to focus on DSB17 data, image pipeline
- 2017-2-22-wed 2nd DL grp meeting: imageJ/nifity/osirix, start to work on dicom metadata ref on pydicom: http://pydicom.readthedocs.io/en/stable/getting_started.html
- 2017-2-23-thu dicom metadata, patient level vs slice level
- 2017-2-24-fri dicom pixel data, dicom geometry
- 2017-2-25-sat conversion, resampling and normalization 
- ... ... w1
- 2017-2-26-sun check training and test sets
- 2017-2-27-mon separate training from test
- 2017-2-28-tue resolve the storage issue by using workspace on server, download data to server per wget + cookies file => review the time line
- 2017-3-1-wed no DL grp meeting, extract the data by 7z, more data from kaggle, revamp MNIST test case on server
- 2017-3-2-thu start to use kernel feature of kaggle /w MNIST
- 2017-3-3-fri ML methodology review: CV, etc
- ... ... w2
- 2017-3-5-sun reading on 3d ct data processing
- 2017-3-6-mon work on the nonsense/idiotic/naive baseline, and tensorflow installation, use new server to run tensorflow
- 2017-3-7-tue reading preprocess and segmentation, kernel test
- 2017-3-8-wed 3rd DL meeting, update doc /w Proj charter, theano broken on "new server"
- 2017-3-9-thu LSF array, install pydicom on server
- 2017-3-10-fri test lsf array for paralle batches
- ... ... w3
- 2017-3-11-sat run on local the "non-sense" model
- 2017-3-12-sun run on server by tf /w more cases, more epoches, and bigger shape, possibly OOM
- 2017-3-13-mon 1st full pass /w silly cnn (66547: 8105.94 sec), correct z-orientation, pre-process all in a batch (67455: 27662.58 sec)
- 2017-3-14-tue LSF array for paralle batch to pre-process tr and test sets, rerun non-sense cnn (95153: 21780.88 sec, 96504: 21772.38 sec)
- 2017-3-15-wed apparently to load mini batches is more time consuming than to load a single npz file, test on this claim, 1 more pass /w single batch (101193: 21520.19 sec), 4th DL grp meeting, Priya joined, Thomas absent
- 2017-3-16-thu spotted a bug in resizing, 128*490*490 drives OOM, 1 pass, test diff shape to maximize the use GPU mem, multi-core not really help
- 2017-3-17-fri mem bottleneck when cnn shape is enlarged /w all samples, able to run on smp node but not gpu node, review the code structure (loop,loop, model), refactoring, start 1-pass /w larger cnn & all samples

#### staged summary: i) Being able to put all the training scans through a 3d cnn smoothly and in a well controled way is a quite non-trivial task; ii) The 3d cnn is not an efficient tool for the diagnosis of lung cancer using ct data

- ... ... w4
- 2017-3-20-mon segmentation 101

## To-do:
- experiment log
- cnn ae: search for "convolutional autoencoder"

## Ref

### DL libs (biased to py):
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
