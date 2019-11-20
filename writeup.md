# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[model]: ./writeup_images/model_plot.png "Model Visualization"
[center_example]: ./writeup_images/center_camera_ex.jpg "Sample training image"
[swerve1]: ./writeup_images/swerve-1.jpg "Recovery Image 1"
[swerve2]: ./writeup_images/swerve-2.jpg "Recovery Image 2"
[swerve3]: ./writeup_images/swerve-3.jpg "Recovery Image 3"
[normal]: ./writeup_images/center_camera_ex.jpg "Normal Image"
[flipped]:  ./writeup_images/center_camera_ex_flipped.jpg "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 filter sizes, reducing to 3x3 in lower layers, and depths between 24 and 64 (model.py lines 81-85). Each convolution layer includes a relu activation layer to allow for non-linearity.

The output of the final convolution layer is flattened and dropout (with keep probability 0.8) is applied.

A Keras lambda is used at the input layer to normalize the data from [0,255] to [-5,5] for zero mean and small variance.

#### 2. Attempts to reduce overfitting in the model

The model contains a dropout layer in order before the first fully connected layer to reduce overfitting (model.py line 93). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code lines 13-14 and 111-115). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 101). The default training rate of 0.001 was used.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I gathered data driving smoothly in the center of the lane, driving the track backwards smoothly in the center of the lane, recovering from either side of the lane back to the center, and finally repeating several sections that the network found difficult.

### Architecture and training documentation

#### 1. Solution Design Approach

Initially I used a simple LeNet style architecture network to get going. It was quickly obvious that was insufficient. Following those initial steps to get the code organized I went back to the class materials and read the NVIDIA blog post "End-to-End Deep Learning for Self-Driving Cars". I decided to replicate the network architecuture (DAVE-2) described there, since it worked for them.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that the was a single dropout layer (with keep probability 0.8) before the fully connected layers.

Initially I had a lot of trouble. The car would often lock onto the right side yellow lane marker and seem to follow it. It would always take the dirt road turnoff after the bridge. Nothing I did helped, not more training data or modifications to the model. Before giving up I reviewed the class and documentation and realized I had forgotten to convert the images from RGB to BGR in the model preprocessing. Once I did this things immediately got a lot better. I suspect it was locking on to the yellow lane marker when running in the wrong color space because that *looked* a lot like the center of the lane. It was trying its best.

Then I retrained and observed improvement in how quickly the training set accuracy.

Here is the final model run output:

```
301/302 [============================>.] - ETA: 0s - loss: 0.0056 - acc: 0.0393Epoch 00001: val_acc improved from -inf to 0.03979, saving model to weights.best.hdf5
302/302 [==============================] - 71s 236ms/step - loss: 0.0056 - acc: 0.0392 - val_loss: 0.0051 - val_acc: 0.0398
Epoch 2/5
301/302 [============================>.] - ETA: 0s - loss: 0.0042 - acc: 0.0392Epoch 00002: val_acc did not improve
302/302 [==============================] - 68s 225ms/step - loss: 0.0042 - acc: 0.0392 - val_loss: 0.0048 - val_acc: 0.0398
Epoch 3/5
301/302 [============================>.] - ETA: 0s - loss: 0.0038 - acc: 0.0393Epoch 00003: val_acc improved from 0.03979 to 0.03979, saving model to weights.best.hdf5
302/302 [==============================] - 69s 228ms/step - loss: 0.0038 - acc: 0.0392 - val_loss: 0.0045 - val_acc: 0.0398
Epoch 4/5
301/302 [============================>.] - ETA: 0s - loss: 0.0034 - acc: 0.0393Epoch 00004: val_acc improved from 0.03979 to 0.04010, saving model to weights.best.hdf5
302/302 [==============================] - 69s 227ms/step - loss: 0.0034 - acc: 0.0395 - val_loss: 0.0045 - val_acc: 0.0401
Epoch 5/5
301/302 [============================>.] - ETA: 0s - loss: 0.0031 - acc: 0.0396Epoch 00005: val_acc did not improve
302/302 [==============================] - 69s 228ms/step - loss: 0.0031 - acc: 0.0396 - val_loss: 0.0042 - val_acc: 0.0400
```

Both the training loss and validation loss improve a bit at each epoch, and validation loss is not out of step with the training loss. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, particularly after the bridge near the dirt road turn off. To improve the driving behavior in these cases, I gathered more data at just those sections, both driving normally and exaggerating avoiding the turn off. I also gathered some more general data driving normally around the track just to ensure I had enough data.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road and for the most part, tracks in the center of the lane.

#### 2. Final Model Architecture

The final model architecture (model.py lines 70-97) consisted of a convolution neural network with the following layers and layer sizes:

| Layer        |  Depth | Filter size, input size, or width |
| ------------- |:-------------:| -----:|
| Input         | 3  | 320x160 cropped to 225x160 |
| Convolution   | 24 | 5x5 |
| Convolution   | 36 | 5x5 |
| Convolution   | 48 | 5x5 |
| Convolution   | 64 | 3x3 |
| Convolution   | 64 | 3x3 |
| Dropout       | N/A  | 2112 |
| Dense     | N/A      |  100 |
| Dense | N/A     |    50 |
| Dense | N/A     |    10 |
| Dense | N/A     |     1 |

Here is a visualization of the architecture taken directly from Keras:

![alt text][model]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded four laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][center_example]

I then recorded driving backwards around the course twice.

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover if it made a mistake and got too close to the edge. Here is a sequence of images from that training (going from the right shoulder back to the center):

![alt text][swerve1]
![alt text][swerve2]
![alt text][swerve3]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles. This was an easy way to generate more generalized data that would help the network not get used to either right or left turns exclusively (depending on the track layout). This doubled the input data quite easily.

![alt text][normal]
![alt text][flipped]

Finally, I recorded several short passes of the section where the bridge ends and the track turns left abruptly. This was because the network had some trouble in that section and I wanted to make sure that section was well represented with good data.

After the collection process, I had 24121 data points. I then preprocessed this data by normalizing, converting from RGB to BGR and creating flipped images.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was between 3-5 as evidenced by the fact that my validation accuracy did not improve much after the first few epochs. I used an adam optimizer so that manually training the learning rate wasn't necessary.
