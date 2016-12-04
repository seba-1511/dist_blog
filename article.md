% An Overview of Distributed Deep Learning
% Seb Arnold
% November 23, 2016

# Introduction
This blog post introduces the fundamentals of distributed deep learning and presents some real-world applications. With the democratization of deep learning methods in the last decade, large - and small ! - companies have invested a lot of efforts into distributing the training procedure of neural networks. Their hope: drastically reduce the time to train large models on even larger datasets. Unfortunately while every commerical product takes advantage of these techniques, it is still difficult for practitioners and researchers to use them in their everyday projects. This article aims to change that by providing a theoretical and practical overview. \newline

Last year, I was lucky to intern at Nervana Systems where I was able to expand their distributed effort. During this 1 year internship, I familiarized myself with a lot of aspects of distributed deep learning and was able to work on topics ranging from implementing efficient GPU-GPU Allreduce routines @opti-mpich to replicating Deepind's Gorila @gorila. I found this topic so fascinating that I am now researching novel techniques for distributed optimization with Prof. [Chunming Wang](http://dornsife.usc.edu/labs/msl/faculty-and-staff/), and applying them to robotic control @comp-trpo-cem with Prof. [Francisco Valero-Cuevas](http://valerolab.org/about/). 

# The Problem 
<!--
    * Introduce formalism and SGD
    * Variants of SGD
    * Tricky points
        * Implementation
        * FC, Convs, and RNNs
        * Benchmarks
-->

## Formulation and Stochastic Gradient Descent

Let's first define the problem that we would like to solve. We are trying to train a neural network to solve a supervised task. This task could be anything from classifying images to playing Atari games or predicting the next word of a sentence. To do that, we'll rely on an algorithm - and its variants - from the mathematical optimization literature: **stochastic gradient descent**. Stochastic gradient descent (SGD) works by computing the gradient direction of the loss function we are trying to minimize with respect to the current parameters of the model. Once we know the gradient direction - aka the direction of greatest increase - we'll take a step in the opposite direction since we are trying to minimize the final error. \newline

More formally, we can represent our dataset as a distribution $\chi$ from which we sample $N$ tuples of inputs and labels $(x_i, y_i) \sim \chi$. Then, given a loss function $\mathcal{L}$ (some common choices include the [mean square error](https://en.wikipedia.org/wiki/Mean_squared_error), the [KL divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence), or the [negative log-likelihood]()) we want to find the optimal set of weights $W_{opt}$ of our deep model $F$. That is,

$$W_{opt} = \arg \min_{W} \mathbb{E}_{(x, y) \sim \chi}[L(y, F(x; W))] $$

\begin{note}
In the above formulation we are not separating the dataset in train, validation, and test sets. However you need to do it !\newline
\end{note} 

In this case, SGD will iteratively update the weights $W_t$ at timestep $t$ with $W_{t+1} = W_t - \alpha \cdot \nabla_{W_t} \mathcal{L}(y_i, F(x_i; W_t))$. Here, $\alpha$ is the learning rate and can be interpreted as the size of the step we are taking in the direction of the negative gradient. As we will see later there are algorithms that try to adaptively set the learning rate, but generally speaking it needs to be chosen by the human experimenter. \newline

One important thing to note is that in practice the gradient is evaluated over a set of samples called the minibatch. This is done by averaging the gradient of the loss for each sample in the minibatch. Taking the gradient over the minibatch helps in two aspects.

1. It can be efficiently computed by [vectorizing](https://goparallel.sourceforge.net/vectorization-feeds-need-speed/) the computations.
2. It allows us to obtain a better approximation of the *true* gradient of $\mathcal{L}(y, F(x; W))$ over $\chi$, and thus makes us converge faster.

However, a very large batch size will simply result in computational overhead since your gradient will not significantly improve. Therefore, it is usual to keep it between 32 and 1024 samples, even when our dataset contains millions of examples.

## Variants of SGD
As we will now see, several variants of the gradient descent algorithm exist. They all try to improve the quality of the gradient by including more or less sophisticated heuristics. For a more in depth treatment, I would recommend [Sebastian Ruder's excellent blog post](http://sebastianruder.com/optimizing-gradient-descent/) and the [CS231n web page](http://cs231n.github.io/neural-networks-3/) on optimization.

### Adding Momentum
Momentum techniques simply keep track of a weighted average of previous updates, and apply it to the current one. This is akin to the momentum gained by a ball rolling downhill. In the following formulas, $\mu$ is the momentum parameter - how much previous updates we want to include in the current one.

#### Momentum
$$ v_{t+1} = \mu \cdot v_t + \alpha \cdot \nabla \mathcal{L}$$
$$ W_{t+1} = W_t - v_{t+1} $$

#### Nesterov Momentum or Accelerated Gradient @nesterov
$$ v_{t+1} = \mu \cdot (\mu \cdot v + \alpha \cdot \nabla \mathcal{L}) + \alpha \cdot \nabla \mathcal{L} $$
$$ W_{t+1} = W_t - v_{t+1} $$

Nesterov's accelerated gradient adds *momentum to the momentum* in an attempt to look ahead for what is coming. 

### Adaptive Learning Rates
Finding good learning rates can be an expensive process, and a skill often deemed closer to art or dark magic. The following techniques try to alleviate this problem by automatically setting the learning rate, sometimes on a per-parameter basis. The following descriptions are inspired by [Nervana's implementation](http://neon.nervanasys.com/index.html/optimizers.html).

\begin{note}
In the following formulas, $\epsilon$ is a constant to ensure numerical stability, and $\mu$ is the decay constant of the algorithm, how fast we decrease the learning rate as we converge.
\end{note}

#### Adagrad @adagrad
$$ s_{t+1} = s_t + (\nabla \mathcal{L})^2 $$
$$ W_{t+1} = W_t - \frac{\alpha \cdot \nabla \mathcal{L}}{\sqrt{s_{t+1} + \epsilon}}$$

#### RMSProp @rmsprop
$$ s_{t+1} = \mu \cdot s_t + (1 - \mu) \cdot (\nabla \mathcal{L})^2 $$
$$ W_{t+1} = W_t - \frac{\alpha \cdot \nabla \mathcal{L}}{\sqrt{s_{t+1} + \epsilon} + \epsilon}$$

#### Adadelta @adadelta
$$ \lambda_{t+1} = \lambda_t \cdot \mu + (1 - \mu) \cdot (\nabla \mathcal{L})^2 $$
$$ \Delta W_{t+1} = \nabla \mathcal{L} \cdot \sqrt{\frac{\delta_{t} + \epsilon}{\lambda_{t+1} + \epsilon}}$$
$$ \delta_{t+1} = \delta_t \cdot \mu + (1 - \mu) \cdot (\Delta W_{t+1})^2 $$
$$ W_{t+1} = W_t - \Delta W_{t+1}$$

#### Adam @adam
$$ m_{t+1} = m_t \cdot \beta_m + (1 - \beta_m) \cdot \nabla \mathcal{L}$$
$$ v_{t+1} = v_t \cdot \beta_v + (1 - \beta_v) \cdot (\nabla \mathcal{L})^2$$
$$ l_{t+1} = \alpha \cdot \frac{\sqrt{1 - \beta_v^p}}{1 - \beta_m^p} $$
$$ W_{t+1} = W_t - l_{t+1} \frac{m_{t+1}}{\sqrt{v_{t+1}} + \epsilon} $$

Where $p$ is the current epoch, that is 1 + the number of passes through the dataset.

### Conjugate Gradients
The following method tries to estimate the second order derivative of the loss function. This second order derivative - the Hessian $H$ - is most ably used in Newton's algorithm ($W_(t+1) = W_t - \alpha \cdot H^{-1}\nabla \mathcal{L}$) and gives extremely useful information about the curvature of the loss function. Properly estimating the Hessian (and its inverse) has been a long time challenging task since the Hessian is composed of $\lvert W \rvert^2$ terms. For more information I'd recommend these papers [@dauphin;@choromanska;@martens] and chapter 8.2 of the deep learning book @dlbook. The following description was inspired by Wright and Nocedal @optibook.

$$ p_{t+1} = \beta_{t+1} \cdot p_t - \nabla \mathcal{L} $$
$$ W_{t+1} = \alpha \cdot p_{t+1} $$

Where $\beta_{t+1}$ can be computed by the Fletcher-Rieves or Hestenes-Stiefel methods. (Notice the subscript of the gradients.)

#### Fletcher-Rieves
$$ \beta_{t+1} = \frac{\nabla_{W_{t}}\mathcal{L}^T \cdot \nabla_{W_{t}}\mathcal{L}}{\nabla_{W_{t-1}}\mathcal{L}^T \cdot \nabla_{W_{t-1}}\mathcal{L}} $$

#### Hestenes-Stiefel
$$ \beta_{t+1} = \frac{\nabla_{W_{t}}\mathcal{L}^T \cdot (\nabla_{W_{t}}\mathcal{L} - \nabla_{W_{t-1}}\mathcal{L})}{(\nabla_{W_{t}}\mathcal{L} - \nabla_{W_{t-1}}\mathcal{L})^T \cdot p_t} $$

# Beyond Sequentiallity

* Introduce sync and async
* Introduce Hogwild + async begets momentum
* Introduce architectures and tricks to make it faster (quantization, ...) (parameter server, mpi, etc...)
* Distributed Synthetic Gradients
* The case of RL: Naive, Gorila, A3C, HPC Policy Gradients

# Benchmarks
* toy problems
* mnist 
* cifar10

# A Live Example

# Acknowledgements

# Citation

# References
Some of the relevant literature for this article. <br />

http://www.benfrederickson.com/numerical-optimization/

http://sebastianruder.com/optimizing-gradient-descent/

http://lossfunctions.tumblr.com/

http://www.denizyuret.com/2015/03/alec-radfords-animations-for.html

https://www.allinea.com/blog/201610/deep-learning-episode-3-supercomputer-vs-pong
