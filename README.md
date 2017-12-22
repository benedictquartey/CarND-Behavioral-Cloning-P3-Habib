# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results in this report


[//]: # (Image References)

[nvidia]: ./examples/cnnArchitecture.png "NVIDIA Model Visualization"
[model]: ./examples/model.png "Model Visualization"
[loss]: ./examples/loss.png "Loss Visualization"
[image2]: ./examples/center.jpg "Center Image"
[image1]: ./examples/left.jpg "Left Image"
[image3]: ./examples/right.jpg "Right Image"
[image5]: ./examples/center-mirror.jpg "Flipped Center Image"
[image4]: ./examples/left-mirror.jpg "Flipped Left Image"
[image6]: ./examples/right-mirror.jpg "Flipped Right Image"
[hist]: ./examples/angleFrequency.png "Histogram of Steering Angles"
[video]: ./examples/thumb.png "Click to Play Video of Car Driving Autonomously"

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* `model.py` containing the script to create and train the model
* `drive.py` for driving the car in autonomous mode
* `model.h5` containing a trained convolution neural network 
* `README.md` summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my `drive.py` file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The `model.py` file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is based on the [NVIDIA model](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/), used by NVIDIA in their end-to-end self driving test. (`model.py` lines 76-96) 

The model includes ELU layers to introduce nonlinearity (code lines 84-92), and the data is cropped and normalized in the model using a Keras lambda layer (`model.py` lines 81-83). 

#### 2. Attempts to reduce overfitting in the model

The model was trained with early termination in order to reduce overfitting (`model.py` lines 104-107). A patience value of 2 was selected for the presented model, however more overfiting could be avoided by setting patience to 1. I tried using dropout and l2 regularization, however they resulted in underfiting. They could be successful in future work if the hyper parameters a better tuned. Below is plot of training and validation loss.

![alt text][loss]

The model was trained and validated on different data sets to ensure that the model was not overfitting (`model.py` line 32). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an ADAM optimizer, so the learning rate was not tuned manually (`model.py` line 102).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a ps3 controller to drive in the center of the lane.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to implement NVIDIA's end-to-end self driving car's steering architecture. I used images from forward facing center, left, and right cameras. I doubled this dataset by flipping the images along the vertical axis.

I used the right and left images as off center data, by adding an offset to the steering angle they gave the model a correction ability.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I then trained the network with on lap of data, and tested in the simulator. My first attempt ran off course after the bridge on the lake level.

To help correct this I ran more laps, and added them to the training set. The model got a little better but was still unable to complete a lap autonomously.

I found that this model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I reduced the number of epochs I trained for. I was originally training for 30 epochs. This meant my model was memorizing my track. I noticed my model's validation loss would stagnate sometimes at 8 epochs, and sometimes at 20. So in order to get maximum validation accuracy I could from the data, I implemented early termination in Keras.

The model was now able to drive the lake track autonomously for 1 lap. However it would ping pong within the lane. At this point I changed the RELU activations into ELU activation, and noticed the regression was much smoother.

The final step was to add data for a second test track in the jungle environment. This resulted in the car being able to almost fully traverse the jungle track. It also smoothed out the turns on the first track. Unfortunately the car still ping pongs on straight portions of the track.

At the end of the process, the vehicle is able to drive autonomously around the lake track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (`model.py` lines 76-96) is based on the [NVIDIA model](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/), used by NVIDIA in their end-to-end self driving test. 

Here is a visualization of the architecture. After every layer I use ELU activation to introduce non linearity. ELU performs much better than the suggested RELU activaiton. Note that in the final model we use the gpu to to crop our images, and use a lambda layer to normalize pixels.

NVIDIA Model (Bottom Up)             |  Final Model
:-------------------------:|:-------------------------:
![alt text][nvidia]  |  ![alt text][model]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on each track trying to drive in the center of the lane with a steady speed. Then I added another lap by driving in the reverse direction. Creating a total of 3 laps per track of training data.

I put 20% of the center images into a validation set. 

For each training image I would also train on the vertically flipped image with a negative steering angle. So If a track had a left turn, we would also train on the same right turn. This helps the model not favor one side over the other. This is shown below.


|        |        Left          |        Center       | Right
|:------:|:--------------------:|:-------------------:|:------------------:
Original | ![alt text][image1]  | ![alt text][image2] | ![alt text][image3]
Steering Angle | 0.98 | 0.78 | 0.58 |
Flipped  | ![alt text][image4]  | ![alt text][image5] | ![alt text][image6]
Steering Angle | -0.98 | -0.78 | -0.58 | 

Notice I also used left and right shifted images. For these images I added a correction factor to the steering angle, so the model also tended towards the middle.

The final training set had 89328 images. Below is a histogram showing a sample of the steering angles of the training data.

![alt text][hist]

We see three prominent peaks. Since most of my steering angles were 0 adding the correction factors to the right and left images creates the peaks at 0.2, and -0.2.

I finally randomly shuffled the data set, and batched them to the train the model.

The validation set helped determine if the model was over or under fitting. It also helped in determine the the ideal number of epochs. The validation loss was used to gauge when we should use early termination. I used an adam optimizer so that manually training the learning rate wasn't necessary.

Below is a video of the simulator on both tracks when being driven by the learned model.

[![alt text][video]](https://youtu.be/oIqQqMipxXU "Video of Car Driving Autonomously")


