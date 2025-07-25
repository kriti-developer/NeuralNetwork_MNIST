# Neural Networks from Scratch (MNIST Project)

**Hello!**

This is my take on building a Neural Network from scratch. It doesn't rely on any machine learning or frameworks. It works with the classic **MNIST dataset** , containing 60,000 training and 10,000 test grayscale images (28 x 28 pixels) of handwritten numbers from 0 to 9.


## 📁 Project Structure
- README.md ( Ctrl + Shift + V for Preview in VS Code )
- Project.py ( Main Python script )
- train.csv ( Training data )
- test.csv ( Unlabeled data to test on )
- sample_submission.csv ( Sample submission file )


## ⚙️ Features Used in This Project
### Activation Functions
- ReLU function ( hidden layers )  
- Softmax ( output layer )
### Loss Function
- Cross Entropy Loss
### Optimization
- Stochastic Gradient Descent ( SGD )
### Initialization
- Random weight and bias initialization ( with He initialization for faster convergence )


## 🧠 How the Network Works
The model consists of three different kinds of layers:
- **Input Layer** containing 784 nodes ( 28x28 pixels flattened )
- **Hidden Layer** ( 2 in this project but can be more ), with ReLU activation
- **Output Layer** containing 10 neurons for each digit, followed by softmax


## 📦 Requirements
- Python 3.x
- NumPy
- pandas
- matplotlib
- scikit-learn

You can install them using:

```bash
pip install numpy pandas matplotlib scikit-learn
```


## ▶️ How to Run
1. Make sure train.csv as well as test.csv are in the same directory.
2. Run the Python script using the command:
```bash
python Project.py
```
3. The script trains the model and prints training loss and validation accuracy per epoch.
4. Later, a submission.csv file is saved in the same directory containing the result of test.csv.


## 📊 Sample Output
```bash
Epoch:  1 Loss:  0.6473019374672335
Epoch:  2 Loss:  0.33479295137982384
Epoch:  3 Loss:  0.28846728866906884
Epoch:  4 Loss:  0.2597563066602134
Epoch:  5 Loss:  0.23690547107774598
Epoch:  6 Loss:  0.21723698718513448
Epoch:  7 Loss:  0.2000129031953101
Epoch:  8 Loss:  0.18505887913282146
Epoch:  9 Loss:  0.17182538689173357
Epoch:  10 Loss:  0.160193488210142
Epoch:  11 Loss:  0.1499711034737352
Epoch:  12 Loss:  0.14080589158048476
Epoch:  13 Loss:  0.13261879445256708
Epoch:  14 Loss:  0.12514591187755844
Epoch:  15 Loss:  0.11833131193505039
Epoch:  16 Loss:  0.11213580867008664
Epoch:  17 Loss:  0.10647998861059171
Epoch:  18 Loss:  0.10124597621102224
Epoch:  19 Loss:  0.09639511427912484
Epoch:  20 Loss:  0.09195475814664733
Epoch:  21 Loss:  0.08786169027025328
Epoch:  22 Loss:  0.08405440135128613
Epoch:  23 Loss:  0.08042526277784005
Epoch:  24 Loss:  0.07705365138299893
Epoch:  25 Loss:  0.0738582313912645
Epoch:  26 Loss:  0.07085812804487718
Epoch:  27 Loss:  0.0680032587721765
Epoch:  28 Loss:  0.06535315044677467
Epoch:  29 Loss:  0.0628169959069811
Epoch:  30 Loss:  0.060426949509431414
Accuracy:  96.15476190476191 %
Submission file saved as: submission.csv
```


## 🧩 Challenges faced
- Tuning learning rate and number of epochs for good accuracy
- Improving random initialisation of weights by He initialisation
- Exploding gradients which ( fixed by clipping delta )
---
### Built By Kriti Dixit (2024B3PS1059G)
