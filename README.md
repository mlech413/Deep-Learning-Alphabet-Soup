# Deep Learning - Alphabet Soup

### Overview
Nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. With machine learning and neural networks, used the features in the dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup. A CSV was processed containing more than 34,000 organizations that have received funding from Alphabet Soup over the years, with a number of columns that capture metadata about each organization.  

Process:
* Using Pandas and scikit-learn’s StandardScaler(), preprocessed the dataset.
* Compiled, trained, and evaluated the neural network model using TensorFlow in which a neural network (deep learning model) was designed to create a binary classification model that predicted if an Alphabet Soup-funded organization will be successful based on the features in the dataset.
* Optimized the model by altering cutoff bins and features, and by building an auto optimizer to help determine activation functions, number of hidden layers, nodes, and epochs.

### Results
Data Preprocessing

Feature "IS_SUCCESSFUL" is the target for the model and indicates if the money was used effectively.
The input features for the model include application type, affiliation, classification, use case, organization, active status, income classification, special considerations, and funding amount requested. All were included in the original model, but the optimized model tested the removal of some of these features.

Compiling, Training, and Evaluating the Model

The original model used two hidden layers with four neurons each, and the "ReLU" activation functions for both. Neurons and layers were kept lower for simple baseline testing on the original model, with 100 epochs run. The "ReLU " activation function was chosen because of its frequent usage and because all of the simplified data is between 0-1, which "ReLU" is designed for. The results were only 72.7% accuracy, short of the desired 75%.

Despite many optimization changes, the optimized model did not fare much better, and did not reach the goal of 75% accuracy. The optimized model used an auto-optimized function to determine possible improvements in the hidden layers, number of nodes, and activation features. The top three results were combined, resulting in 6 hidden layers (1 "ReLU" and five "Sigmoid"), with nodes ranging from 5 to 200. The epochs were averaged and resulted in only 14. In addition, the cutoff bin for the classification feature was adjusted downward from 1000 to 500, and the status feature was removed. These changes were made after numerous trial-and-error runs showed they yielded some slight improvements. The end result was a disappointing 73% accuracy.

### Summary
The overall results of the deep learning model did not meet expectations. The combining of auto optimization results did not improve results significantly. Additional testing is needed to focus and fine-tune a single auto optimized result, rather than attempting to incorporate all of the top three.  

In addition to changes to the currently used model, I would recommend trying some other types of models or algorithms.  In particular, a random forest algorithm has some interest in this area. In this algorithm, a series of simple trees randomly samples the data and creates a decision tree for only that small portion of data. When they are combined, many slightly better than average small decision trees can create a strong classifier that has much better decision-making power. It reduces overfitting because each weak classifier is trained on different pieces of the data, it works well with outliers and non-linear data, and runs efficiently on large databases.
