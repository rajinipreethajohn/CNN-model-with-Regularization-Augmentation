**CNN Project for CIFAR10 Dataset**
This project is a Convolutional Neural Network (CNN) implementation on the CIFAR10 dataset. The goal of this project is to achieve high accuracy on both train and test datasets using regularization and augmentation techniques.

**Dataset**
The CIFAR10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. There are 50,000 training images and 10,000 test images.

**Model Architecture**
We create a Sequential model and start adding layers one by one. The first Conv2D layers are preceeded by MaxPooling2D and Dropout layer. Then 3 Conv2D layers are stacked followed by again a pooling and dropout layer followed by 2 fully connected Dense layers leading to an output layer. The kernel_size and pool_sie are the same through out the network.

The filters double in the size with every layer starting from 128 going up to 512 and coming back down to 256 in the fifth layer. Similar values for neurons were used in the fully connected layer. These settings gave me the best accuracy though it can be computationally expensive. With the help of a GPU on Google Colab, I was able to train this model in 47 mins, 32 secs. I went with the standard activation 'relu' and 'same'padding'

**Regularization and Augmentation**
To prevent overfitting, I have used the l2 kernel_regularizer and also added dropout layers. This reduced overfitting while also icreasing accuracy by a few percent. I again exprimented with a varity of dropout values to land on this one, using lower dropout of 0.3 for the conv layers and a higher 0.5 for the fully connected layers.

**Training**
The model is trained with a lr=0.0003, decay=1e-6 and a batch size of 64. The model is trained for 125 epochs on the training set, and the accuracy is evaluated on both the training and validation sets after each epoch.

**Results**
After training for 125 epochs, the model achieves an accuracy of 85% on both the train and test datasets, which indicates that the model is not overfitting to the training data. The accuracy on the test set is impressive, considering that the model uses a CNN architecture with regularization and augmentation.

**Future Work**
In the future, we could try more complex model architectures such as ResNet or DenseNet to achieve even higher accuracy. We could also experiment with different regularization and augmentation techniques to further improve the performance of the model.

**Requirements**
Python 3.x
TensorFlow 2.x
NumPy
Matplotlib
Pillow
