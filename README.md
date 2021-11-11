# Handcrafting-MLP-using-NumPy



In this notebook series, we will create a single hidden-layer MLP for performing **Binary Classification** on a synthetic dataset. The backpropagation algorithm is implemented using the batch Gradient Descent. 

We will **use only the NumPy python library** for creating the MLP model. For the implementation, the notebooks provide pseudocode description of the backpropagation algorithm. A user will have to provide the NumPy implemention.


## MLP Architecture

The MLP consists of 3 layers.
- First layer: input layer (it should have two "neurons" since the input data is 2D)
- Second layer: hidden layer (it should have 4 neurons)
- Third layer: output/classification layer (it should have only one neuron since it's a binary clasification problem)


There are 3 notebooks in this repository.
- Notebook 1 and 2 use the mean squared error (MSE) as the loss function. Although MSE is not an ideal loss function for classification problems, we use it for its convenience of implementation. 
- Notebook 3 uses the binary cross-entropy as the loss function.

The main difference between notebook 1 and 2 is in their implementation of the bias weights.


    Notebook 1: MLP-Handcrafting-I


##### Bias weights
Neurons in the hidden layer and the final layer will have a bias weight. For example:
- Hidden layer: the 4 neurons in the hidden layer will have 4 bias weights. 
- Final layer: the single neuron in the final layer will have a bias weight. 

These bias weights are added as a separate row in the weight matrices for the hidden layer and the final layer. Consequently, both in the hidden layer and in the final layer, a bias neuron is added with the feature neurons. This is done by adding a column of 1s with the input data and the hidden layer activation signal data.



    Notebook 2: MLP-Handcrafting-II-Separate Bias Calculation


##### Bias weights
The bias weights for the hidden layer and the final layer neurons are **computed separately**.
- Hidden layer: each of the 4 neurons in the hidden layer will have a bias weight. Thus, the hidden layer bias weight should be a (4 x 1) array
- Final layer: the single neuron in the final layer will have a bias weight. Thus, the final layer bias weight should be a (1 x 1) array



    Notebook 3: MLP-Handcrafting-III-Cross-Entropy Loss
   
Unlike the previous two notebooks, we will use the **binary cross-entropy** as the loss function. The model should converge faster than the previous two cases.

The bias weights will be implemented similar to notebook 1.


