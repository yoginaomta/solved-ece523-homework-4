Download Link: https://assignmentchef.com/product/solved-ece523-homework-4
<br>
<ul>

 <li>Multi-Layer Perceptron</li>

</ul>

In class we discussed the derivation of the backpropagation algorithm for neural networks. In this problem, you will train a neural network on the CIFAR10 data set. Train a Multi-Layer Perceptron (MLP) neural network on the CIFAR10 data set. This is an opened implementation problem, but I expect that you implement the MLP with at least two different hidden layer sizes and use regularization.

<ul>

 <li>Report the classification error on the training and testing data each configuration of the neural network. For example, you should report the results in the form of a table</li>

</ul>

<table width="348">

 <tbody>

  <tr>

   <td width="189"> </td>

   <td width="92">Classificati training</td>

   <td width="66">on Error testing</td>

  </tr>

  <tr>

   <td width="189">50HLN+no regularization</td>

   <td width="92">0.234</td>

   <td width="66">0.253</td>

  </tr>

  <tr>

   <td width="189">50HLN+<em>L</em><sub>2 </sub>regularization</td>

   <td width="92">0.192</td>

   <td width="66">0.203</td>

  </tr>

  <tr>

   <td width="189">250HLN+no regularization</td>

   <td width="92">0.134</td>

   <td width="66">0.153</td>

  </tr>

  <tr>

   <td width="189">250HLN+<em>L</em><sub>2 </sub>regularization</td>

   <td width="92">0.092</td>

   <td width="66">0.013</td>

  </tr>

 </tbody>

</table>

List all the parameters that you are using (i.e., number of learning rounds, regularization parameters, learning rate, etc.)

<ul>

 <li>I would suggest using Google’s TensorFlow, PyTorch or Keras library to implement the MLP; however, you are free to use whatever library you’d like. If that is the case, here is a link to the data</li>

 <li>I recommend using a cloud platform such as Google Colab to run the code.</li>

</ul>

<ul>

 <li>Adaboost [20pts]</li>

</ul>

Write a class that implements the Adaboost algorithm. Your class should be similar to sklearn’s in that it should have a fit and predict method to train and test the classifier, respectively. You should also use the sampling function from Homework #1 to train the weak learning algorithm, which should be a shallow decision tree. The Adaboost class should be compared to sklearn’s implementation on datasets from the course Github page.

<ul>

 <li>Recurrent Neural Networks for Languange Modeling</li>

</ul>

Read “LSTM: A Search Space Odyssey” (<a href="https://arxiv.org/abs/1503.04069">https://arxiv.org/abs/1503.04069</a><a href="https://arxiv.org/abs/1503.04069">)</a>. One application of an RNN is the ability model language, which is what your phone does when it is predicting the top three words when you’re texting. In this problem, you will need to build a language model.

You are encouraged to start out with the code <a href="https://www.tensorflow.org/tutorials/text/text_generation">here</a><a href="https://www.tensorflow.org/tutorials/text/text_generation">.</a> While this code will implement a language model, you are required to modify the code to attempt to beat the baseline for the experiments they have implemented. For example, one modification would be to train multiple language models and average, or weight, their outputs to generate language. Write a couple of paragraphs about what you did and if the results improve the model over the baseline on Github.