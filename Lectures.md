# Learning Systems DT8008

# 2020:01:23

## **What is machine learning?**
**Definition:**
*   A computer is said to learn from experience E with respect to some task T and performance measure P if it's performance on T,as measured by P improves with experience E.

### **Supervised Learning**
Intro for regression and classification
*   Regression: Estimation of a continuos function
*   Classification: Estimation of discrete values

### **Unsupervised Learning**
In **unsupervised** learning (e.g. clustering) we have an **unlabeled** training set.


### **Data Representation**
 Features are denotes as $x$
### **Model Representation**
*   The model (to be learned) is a function $h$ (called hypothesis)
*   The model has parameters $\theta_0$,$\theta_1$,..
*   $\theta$ = ($\theta_{0},\theta_{1}$,...) is the vector of parameters, so the model is denoted $h_{\theta}$
* Learning means finding the optimal parameters on a given dataset
* The output value is computed through $h_{\theta}(x)$ = $\vec{\theta} \bullet \vec{x}$
* The error $E(\theta)$ of a model is calculated through:
* $E(\theta)=(\theta) = \sum_{i=1}^{n}{[ h_{\theta}(x^{i}) - y^{i}]^{2}}$
  * Where $h_{\theta}(x^{i})$ is the predicted and $y^i$is the true output for $x^i$
* The error is also called the **loss** or **cost**


___

# 2020:01:27

## **Regression**

### **Linear Regression**
#### With one feature
A linear model **$h_{\theta} = \theta_{0} + \theta_{1}x$** is to be learned for the dataset. 
In linear regression, parameters $\theta_{0}$ and $\theta_{1}$ are decided through minimizing the error function through:
**$\min_{\theta_{0},\theta_{1}}\frac{1}{2n}E(\theta)=(\theta) = \sum_{i=1}^{n}{[ h_{\theta}(x^{i}) - y^{i}]^{2}}$** This is done through finding minima of the derivative of the error function $E(\theta)$. 

This optimization algorithm is called **Gradient Descent**. Which is a general optimization algorithm, not only specific for this error function. 

## **Gradient Descent**
**Gradient Descent** is a algorithm which finds a local minimum e.g. a error function $E(\theta_{0},\theta_{1})$ by computing the gradient $\nabla E(\theta_{0},\theta_{1})$ and updating the parameters  $\theta_{0}$ and $\theta_{1}$ until a minimum is found.

Since Gradient Descent will find a local minimum, it might not find the optimal solution without a added method. For example starting with several random parameter values.


Repeat untill convergence:
### $\theta_{j} \lArr \theta_{j} - \alpha \frac{\delta E}{\delta \theta_{j}}$
Where $\alpha$ is the **learning rate** scalar.

**Check online** Gradient Descent with momentum

### Gradient Descent vs. Normal Equation
| Gradient Descent  | Normal Equation |
|-------------------|-----------------|
| need to choose $\alpha$| No need to choose $\alpha$|
| Needs several iterations| check slide|
| check slide | check slide |




## **Non-linear regression**
Regression can also be used for non-linear functions,  $h_{\theta}(x) = \theta_{0} + \theta_{1}x + \theta_{2} x^{2} + ... + \theta_{n} x^{n}$. 

### **K-Nearest-Neighbors (KNN) for Non-linear Regression**
There are KNN with uniform weights and KNN with different weights.

### **Kernel Regression**
*   Very similar to the weighted KNN method
*   A common kernel regression model is the Nadaraya-Watson estimator, with a **Gausian kernel function**

$k(\vec{u},\vec{v}) = e^{-||\vec{u}-\vec{v}||^2 / 2\sigma ^{2}}$

with $\gamma = \frac{1}{\sigma ^2}$, $k(\vec{u},\vec{v}) = e^{-\gamma || \vec{u} - \vec{v}||^2}$

$h(x) =\frac{1}{\sum_{i=1}^{n} k(x,x^{(i)})} \sum_{i=1}^{n}k(x,x^{(i)}y^{(i)})$


___


# 2020:02:06

## **Classification**

### **Logistic Regression**
*   Classification method despite its name
*   In a binary classification we want $y = 0$ or $y = 1$
    *   but if you use a simple linear regression model $h_{\theta}(x) = \theta^{T}x$,then $h_{\theta}(x)$ can be $>1$ or $<0$
*   The logistic regression model is defines so that 0 < h_{}... read slides
* Error function : $E(\theta) = \frac{1}{n}\sum{cost}$ Check slides

# 2020-02-11

## **Overfitting and Generalization**
* **Overfitting**
  * Low Bias & High Variance
* **Underfitting**
  * High Bias & Low Variance

The problem of overfitting occurs when a model is too adjusted for the training set. 
In regression, overfitting/underfitting leads to drastically incorrect estimated values. In classification, it leads to bad estimation of classes.
### What makes it more likely to overfit
*   Not enough training examples ( small trainign dataset)
*   Too many features
*   Using a non-convinient type of models/hypothesis functions (e.g. too much complex for our problem/data)


### Generalization Error
* The training error is the error of the model $h_{\theta}$ on the training examples
  * It is not an estimation of the error that the model $h_{\theta}$ will have when deployed and applied on new data
* The **generalizaiton error** is the error of the model $h_{\theta}$ on new (unseen) data
  * The generalization error is typically higher than the training error
* How do we estimate the generalization error of some model?
  * Using k-fold Cross Validation (**k-CV**)
  * Using Leave-One-Out estimate (**LOO**)
  


# **2020:17:02**

