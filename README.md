# Richtige MeinMNIST Netz

This project contains a neural network implemented in TensorFlow to classify handwritten digits from the **MNIST dataset**. The notebook **Richtige_MeinMNIST_NETZ.ipynb** explores a custom-built neural network architecture, demonstrates the training and evaluation process on the MNIST dataset, and provides compilation and deployment instructions using **TVM** and **microTVM** for embedded systems.

## Features

- **Custom Neural Network Architecture**: Built from scratch using TensorFlow/Keras.
- **Digit Classification**: Achieves high accuracy on the MNIST dataset.
- **Quantization and Deployment**: Integrates TVM and microTVM for optimizing and running the model on microcontrollers.
- **Efficient Inference**: Leverages TVM for compiling models into optimized code for microcontrollers and edge devices.

## Getting Started

### Prerequisites

- Python 3.x
- Jupyter Notebook or Google Colab
- TensorFlow 2.x
- TVM and microTVM
- Required libraries:
  - numpy
  - matplotlib
  - sklearn
  - tvm

You can install the necessary Python packages using the following:
```bash 
pip install tensorflow numpy matplotlib scikit-learn tvm
```

### Dataset
The MNIST dataset is used in this project. It consists of 70,000 grayscale images of handwritten digits (28x28 pixels) from 0 to 9. The dataset is split into 60,000 training images and 10,000 testing images.

The dataset is automatically loaded via TensorFlow’s ```tf.keras.datasets``` module.

### Installation

1. **Clone the repository**:
``` bash
git clone <repository_url>
cd <repository_directory>
```

2. **Run the notebook**: You can either run the notebook locally using Jupyter Notebook or on Google Colab:

  * Local Jupyter Notebook:
  ``` bash
  jupyter notebook Richtige_MeinMNIST_NETZ.ipynb
  ```
  * Google Colab: Upload the notebook to Colab and run it there for easy access to GPU/TPU support.

### Training the Model

1. **Model Architecture**:

The neural network is composed of fully connected (Dense) layers, activation functions (ReLU, Softmax), and dropout for regularization.

2. **Training**:

The model is trained using the ````Adam```` optimizer and categorical cross-entropy loss.
The training process displays accuracy and loss metrics during each epoch.

3. **Evaluation**:

The model is evaluated on the test set, and performance metrics, including accuracy and confusion matrices, are provided.

## TVM and microTVM Compilation
To deploy the trained model on embedded devices using TVM and microTVM, follow these steps:

1. **TVM Setup**
TVM enables the compilation of deep learning models into highly optimized code for various hardware targets. To use TVM, first ensure it is installed and properly configured:

* Install TVM using the following:
```` bash 
git clone https://github.com/apache/tvm
cd tvm
mkdir build
cp cmake/config.cmake build/config.cmake
# Customize the config.cmake for your environment
cd build
cmake ..
make -j4
````
* Install the Python bindings:
```` bash 
cd python
pip install -e .
````
2. **Model Conversion with TVM**
After training, convert the TensorFlow model into a format suitable for TVM:
```` Python
import tensorflow as tf
import tvm
from tvm import relay
from tvm.contrib import graph_executor

# Load the saved Keras model
model = tf.keras.models.load_model('mnist_model.h5')

# Convert the model into TVM Relay IR
shape_dict = {"input_1": (1, 28, 28, 1)}
mod, params = relay.frontend.from_keras(model, shape_dict)

# Set target hardware (e.g., ARM Cortex-M for microcontrollers)
target = tvm.target.Target("llvm")

````

3. **microTVM Compilation**
To deploy the model on a microcontroller using microTVM:

* Set the target to an embedded platform (e.g., Cortex-M):
````python
target = tvm.target.target.micro("host")
````
* Compile the model for the microcontroller:
````python
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)
````
4. **Flash to Microcontroller**
To deploy the compiled model on a device:

* Connect your microcontroller (e.g., D1 Mini or STM32).
* Use the TVM RPC server to load and run the model on the device:
````python
from tvm.micro import create_local_graph_executor, Session

session = Session(device_config="openocd")  # adjust for your hardware
graph_mod = create_local_graph_executor(lib["default"](session))

# Run inference
graph_mod.set_input("input_1", test_input)
graph_mod.run()
output = graph_mod.get_output(0)
````
5. **Running on microTVM**
For more detailed instructions on using microTVM, refer to the official TVM microTVM guide.

### Quantization
This notebook also implements quantization techniques (such as Float16 Quantization) for reducing model size and improving performance on edge devices.

### Results
The model achieves high accuracy on both the training and test datasets.
Using TVM and microTVM, the model can be efficiently deployed on low-resource microcontrollers while maintaining reasonable performance.
Customization
You can modify the neural network architecture, experiment with different optimizers, and adjust quantization strategies to fit the deployment constraints on your target hardware.

### Contributing
If you have any suggestions or improvements, feel free to submit a pull request or open an issue.

### License
This project is licensed under the MIT License - see the LICENSE file for details.

````javascript

You can copy and save this as `README.md` in your project directory. Let me know if you need further changes!

````