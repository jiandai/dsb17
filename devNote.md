
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
- 2017-3-11-sat 1st sampled pass on local /w "non-sense" model : (20, 1, 128, 128, 128), Train on 18 samples, validate on 2 samples, Epoch 1/1, 18/18 [==============================] - 4869s - loss: 0.6869 - acc: 0.7222 - val_loss: 7.5440 - val_acc: 0.0000e+00
- 2017-3-12-sun run on server by tf /w more cases, more epoches, and bigger shape, possibly OOM
- 2017-3-13-mon 1st full pass /w silly cnn (job 66547: 8105.94 sec, (1397, 1, 128, 128, 128), on 10th epoch 1257/1257 [==============================] - 525s - loss: 0.5716 - acc: 0.7430 - val_loss: 5.3991 - val_acc: 0.6286), correct z-orientation, pre-process all in a batch (job 67455: 27662.58 sec)
- 2017-3-14-tue LSF array for paralle batch to pre-process tr and test sets, rerun non-sense cnn (job 95153: 21780.88 sec, (1397, 1, 128, 128, 128), on 10th epch: 1257/1257 [==============================] - 2138s - loss: 0.5731 - acc: 0.7430 - val_loss: 10.4118 - val_acc: 0.3429) (job 96504: 21772.38 sec, (1397, 1, 128, 128, 128), on 10th epoch: 1257/1257 [==============================] - 2137s - loss: 0.5724 - acc: 0.7430 - val_loss: 9.8844 - val_acc: 0.3571)
- 2017-3-15-wed apparently to load mini batches is more time consuming than to load a single npz file, test on this claim, 1 more pass /w single batch (job 101193: 21520.19 sec, (1397, 1, 128, 128, 128), on 10th epoch: 1257/1257 [==============================] - 2135s - loss: 0.5716 - acc: 0.7430 - val_loss: 6.2091 - val_acc: 0.5357), 4th DL grp meeting, Priya joined, Thomas absent
- 2017-3-16-thu spotted a bug in resizing, 128*490*490 drives OOM, 1 full pass (job 209894, (1397, 1, 128, 256, 256), on 10th epoch: 1257/1257 [==============================] - 1954s - loss: 0.5722 - acc: 0.7430 - val_loss: 7.2016 - val_acc: 0.5000), test diff shape to maximize the use GPU mem, multi-core not really help
- 2017-3-17-fri mem bottleneck when cnn shape is enlarged /w all samples, able to run on smp node but not gpu node, review the code structure (loop,loop, model), refactoring, start 1st dry-run /w larger cnn & all samples (job 423878, (1397, 1, 160, 300, 300), on 10th epoch: 1257/1257 [==============================] - 15329s - loss: 0.5715 - acc: 0.7430 - val_loss: 8.8406 - val_acc: 0.4143)

#### staged summary: i) Being able to put all the training scans through a 3d cnn smoothly and in a well controled way is a quite non-trivial task; ii) The 3d cnn is not an efficient tool for the diagnosis of lung cancer using ct data

- ... ... w4
- 2017-3-20-mon segmentation 101
- 2017-3-21-tue 2.5-mm resolution dry-run result out (job 452714, (1397, 1, 171, 196, 196), on 10th epoch: 1257/1257 [==============================] - 6991s - loss: 0.5714 - acc: 0.7430 - val_loss: 7.9096 - val_acc: 0.4429)
- 2017-3-22-wed 5th DL meeting /w bigger team, re-baseline to check cuda dependency (theano broke), clone official tutorial, met /w cleo`s team, sample test using pretrained unet without ROI, 3d cnn add normalization for 3rd dry-run (job 520787, (1397, 1, 171, 196, 196), on 10th epoch: 1257/1257 [==============================] - 6993s - loss: 0.5724 - acc: 0.7430 - val_loss: 0.5955 - val_acc: 0.7214)

## To-do:
- add more preproc in m0
- add ROI /w pretrained unet
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
