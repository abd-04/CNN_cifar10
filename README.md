 CIFAR-10 Image Classification Using Convolutional Neural Network
Project Report
By Abdullah Owais
________________________________________
1. Introduction 
I started studying Machine learning during the summer and Neural networks caught my eye, they seemed interesting as most of the “AI” we use daily is based on neural networks. All of these Large Language Models fundamentally rely on Neural Network architecture. So, to deepen my understanding about the concept even more I started off with classifying the cifar10 dataset using just neural network’s dense layers, with the activations. The accuracy I was getting was around 67%, test accuracy. And that’s not a good result for any image classification. 
I solved this problem by utilizing the Convolutional architecture. The NN performs well on images where there is less relation between each pixel, it treats each pixel equally so it fails to understand relation between each pixel. It fails where spatial features matter. Using ONLY dense layers flatten images into 1D vectors where each value in the vector row is treated equally which fails tasks where pixel correlation matters (most images).
________________________________________
2. The Solution – CNN Architecture
The problem is not having spatial info from the images. This is solved by adding another layer before the fully connected dense layers, called the convolutional layer.
CNN’s are designed to process grid data like images where relationship between pixels carry meaningful info, such as edges, curves, objects in an image. They need to be detected first before classifying the image as whatever label provided. The flow of a simple NN is below:
Image → Flatten → Fully-Connected Layers → Activation → Softmax Output
Here flattening the image in the beginning destroys any chances of gaining spatial characteristics about the image. In a CNN, we have multiple convolutional layers before we flatten image into a dense connected layer. In those conv layers, the network learns multiple features such as curves, edges on the image. The flow is like this:
Image → Conv layer(learn spatial patterns i.e edges) → pooling layer(compressing spatial features) → More convolutions(deeper patterns i.e shapes) → Flatten(convert learnt feature maps into a row vector) →fully connected layers→ Output
Flattening the image before destroys the structure and when we apply it in CNN, we basically have the features detected already in a grid called the feature map, then it is flattened and fed into the dense Neural Network. 
The feature map is calculated by doing a convolution operation between a small patch of the image and a filter or a kernel. The filter is a 3x3 matrix that is used to detect patterns like vertical, horizontal and diagonal edges. The filter slides over 3x3 patches of the image and the conv operation is applied.

Pooling layers then reduce the spatial dimensions of these feature maps while preserving the important information. This reduces computation and helps capture more abstract features at deeper layers. The number of filters increases with depth (for example, 32 → 64 → 128), allowing the network to learn more complex representations.
________________________________________
3. Dataset Description
3.1 CIFAR-10 Overview
•	60,000 images
•	32×32 resolution
•	10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
3.2 Data Splitting
Total 60,000 images that are split into train and testing data, I made the validation set too by splitting the training data more.
•	50,000 training images
•	5,000 validation images
•	10,000 test images
3.3 Preprocessing Steps
•	Normalization (divide by 255)
•	One-hot encoding
•	Data augmentation  Makes the model see different transformations of the same image to help increase generalization
o	rotation
o	width/height shift
o	horizontal flip
o	random zoom

________________________________________

4. Model Architecture
There are several well-known CNN architectures, such as VGG, ResNet, and Inception. The model I implemented is built from scratch, but its structure closely resembles a VGG-style network. I put a total of 6 layers in my network. We do not count pooling layers as part of trainable parameters. I will refer a group of convolutions and a pooling block as a                ‘Conv Block’. A total of 4 Conv layers and 2 dense layers are used in this network. The architecture mimics a VGG network in the sense that each block contains two convolution layers followed by a max-pooling layer. The flow of my model is as follows:

Input (32×32×3)
→ Conv(32x32x32) → BN → ReLU
→ Conv(32x32x32) → BN → ReLU
→ MaxPool(16x16x32) → Dropout
→ Conv(16x16x64) → BN → ReLU
→ Conv(16x16x64) → BN → ReLU
→ MaxPool(8x8x64) → Flatten→ Dense(512) → Dense(10) → Softmax 
Pooling reduces the height and the width by half, and after every conv block the number of filters increases.
 

5. Model Summary


 ________________________________________
6. Training Methodology
•	 Optimizers Used: Adam (lr = 0.0003)
•	 Loss Function: Categorical Crossentropy
•	 Callbacks: EarlyStopping
•	 Hyperparameters
o	Batch size: 96
o	Epochs: 60 (early stopping)
o	Dropout: 0.25, 0.5
o	Batch Normalization after every conv/dense block
________________________________________

7. Results and Analysis
7.1 Training and Validation Accuracy:
  
The curves indicate that the model is learning well, with the training and validation performance remaining close to each other. This suggests that the model is not overfitting and is generalizing effectively to unseen data.

 
The model stopped training at epoch 55 through early stopping, achieving a validation accuracy of 81.9% and a validation loss of 0.5923, indicating stable performance and good generalization.
7.2 Final Model Performance:
 
Final Test Accuracy: 83%, Test Loss: 0.494________________________________________
8. Final Remarks
The model achieved over 83% accuracy using a VGG-style CNN with Batch Normalization, Dropout, and Data Augmentation.
________________________________________
9. Future Enhancement
I can add another conv layer too to make it a deeper VGGnet, hopefully it shall push the test accuracy upto 85%.
________________________________________

