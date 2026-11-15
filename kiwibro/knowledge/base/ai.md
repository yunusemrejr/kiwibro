# AI and machine learning

## Artificial intelligence (AI, machine intelligence)
Artificial intelligence is the field of building computer systems that do tasks which normally need human intelligence, such as recognising speech, understanding language or planning moves in a game. Machine learning is now the most common route to it.

## Machine learning (ML, learning from data)
Machine learning is a branch of AI in which a program learns its behaviour from example data instead of following rules written by hand. Shown many labelled photos of cats and dogs, it adjusts its internal numbers until it can tell the two apart in photos it has never seen.

## Deep learning (deep neural networks)
Deep learning is machine learning with neural networks that have many layers stacked one after another. The extra layers let the network build up more abstract features, from edges in an image to shapes and then whole objects.

## Supervised learning (labelled data)
Supervised learning trains a model on examples that already have the correct answer attached, called labels. The model learns to predict the label for new inputs, as in a spam filter trained on emails marked spam or not spam.

## Unsupervised learning (no labels, finding structure)
Unsupervised learning finds structure in data that has no labels, such as groups of similar customers or topics hidden in a set of documents. Nobody gives the model the right answers; it looks for patterns in the inputs alone.

## Reinforcement learning (reward and penalty)
Reinforcement learning trains an agent to make a sequence of decisions by rewarding good outcomes and penalising bad ones. Through trial and error it learns a strategy, such as how to play a board game to win more often.

## Training vs inference (learning vs running a model)
Training is the process of adjusting a model's weights using data, which is slow and needs a lot of computing. Inference is using the finished, fixed model to produce an output for a new input, which is much cheaper for each request.

## Dataset (training data, examples)
A dataset is the collection of examples used to build or test a machine learning model, such as thousands of labelled images or a table of customer records. The quality and variety of the dataset limit what the model can learn.

## Training, validation and test split (data split)
A training, validation and test split divides a dataset into three parts. The training set teaches the model, the validation set guides choices such as settings and when to stop, and the test set is kept aside to give a final score the model has never been tuned on.

## Overfitting (memorising training data)
Overfitting happens when a model learns its training examples too closely, including their noise and accidents, so it scores well on training data but poorly on new data. A large gap between training accuracy and test accuracy is a common sign.

## Underfitting (model too simple)
Underfitting happens when a model is too simple, or was trained too little, to capture the real pattern in the data. It performs poorly even on the training data, and a more flexible model or longer training can help.

## Regularisation (preventing overfitting)
Regularisation is any technique that discourages a model from becoming too complicated, so that it generalises better to new data. A common form adds a penalty to the loss for large weights, which stops the model from relying heavily on any single input.

## Dropout (neural network regularisation)
Dropout is a regularisation method for neural networks in which random neurons are switched off at each training step. This stops the network from depending on particular neurons, and it is turned off when the trained model is used.

## Loss function (cost function, error measure)
A loss function measures how wrong a model's prediction is compared with the correct answer, and gives the result as a single number. Training tries to make this number smaller; mean squared error, for example, averages the squared differences between predictions and answers.

## Gradient descent (how models learn)
Gradient descent is the basic method for training neural networks. It measures how the loss changes as each weight changes, then moves every weight a small step in the direction that lowers the loss, and repeats this many times.

## Learning rate (step size)
The learning rate is the size of each step that gradient descent takes when it updates the weights. A value that is too large makes training overshoot or diverge, and a value that is too small makes training very slow.

## Batch size (mini-batch)
Batch size is the number of examples processed together before the model updates its weights once. Larger batches need more memory and give smoother updates, while smaller batches update more often and fit into less memory.

## Epoch (training pass)
An epoch is one complete pass through the whole training dataset. Training usually runs for many epochs, and the validation score is often checked after each one to decide when to stop.

## Backpropagation (backward pass, gradients)
Backpropagation is the method that works out how much each weight in a neural network contributed to the error. It passes the error backwards from the output layer to the input layer, and the results are the gradients that gradient descent uses to update the weights.

## Neural network (artificial neural network)
A neural network is a model made of layers of simple units called neurons, where each layer passes its numbers on to the next. Adjusting the weights between the layers lets the network map an input, such as the pixels of an image, to an output, such as a label.

## Neuron (artificial neuron, node)
An artificial neuron multiplies each of its inputs by a weight, adds the results together with a bias, and passes the total through an activation function. It is a simplified model of a nerve cell, and a neural network is built from many of them.

## Weights and biases (model parameters)
Weights are the numbers a neural network adjusts during training to set how strongly each input affects a neuron. A bias is an extra number added to a neuron's total so that it can shift its output, and together the weights and biases are the parameters the model learns.

## ReLU (rectified linear unit, activation)
ReLU is an activation function that outputs its input when the input is positive and zero otherwise, written as max(0, x). It is simple and fast to compute, and it is a common choice inside the hidden layers of modern networks.

## Sigmoid (logistic function, activation)
Sigmoid is an activation function that squeezes any number into a value between 0 and 1 along an S-shaped curve. It is useful for producing a probability, though deep networks often use other activations inside their hidden layers.

## Tanh (hyperbolic tangent, activation)
Tanh is an activation function that squeezes any number into a value between -1 and 1 along an S-shaped curve. Its output is centred on zero, which is why it is often used where negative values are needed, such as inside some recurrent networks.

## Softmax (probabilities from scores)
Softmax turns a list of raw scores into probabilities that each lie between 0 and 1 and add up to 1. It is usually the last step of a classifier, so the largest score gets the largest probability.

## Perceptron (single neuron classifier)
A perceptron is the simplest neural network: one neuron that takes weighted inputs and outputs one of two classes by comparing the total with a threshold. A single perceptron can only learn problems whose two classes can be separated by a straight line.

## Convolutional neural network (CNN, image network)
A convolutional neural network slides small learned filters across an image, so that each filter detects the same kind of pattern, such as an edge, wherever it appears. This makes it efficient for images and it is widely used in computer vision.

## Recurrent neural network (RNN, sequence model)
A recurrent neural network processes a sequence one item at a time and keeps a hidden state that carries information forward. It suits data that comes in order, such as text or sound, but a basic version struggles to remember information from far back in a long sequence.

## LSTM (long short-term memory)
An LSTM is a type of recurrent network that uses gates to decide what to keep, what to forget and what to output from its memory. The gates help it remember information across longer sequences than a basic recurrent network can.

## Transformer (attention architecture)
A transformer is a neural network design that processes all the items of a sequence together using attention, instead of one item at a time. Most modern large language models are built on this design.

## Attention (attention mechanism)
Attention lets a model weigh how much each part of its input matters when it produces each part of its output. When translating a word, for example, the model can focus on the most relevant words of the source sentence.

## Self-attention (token attention)
Self-attention is attention applied within a single sequence, so that every token looks at every other token in that same sequence to decide which context matters. It is the core operation inside a transformer.

## Encoder vs decoder (transformer parts)
An encoder reads an input and turns it into a set of numerical representations, while a decoder writes an output one token at a time from those representations. Some models use only an encoder to understand text, some use only a decoder to write text, and translation models often use both.

## Embedding (vector embedding, word vectors)
An embedding is a list of numbers that represents an item, such as a word, sentence or image, so that items with similar meaning get similar numbers. Comparing these lists lets a computer measure how alike the meanings are.

## Tokenizer (text tokenizer, tokenisation)
A tokenizer splits text into tokens and converts each token into a number, which is the form a model can process. The same tokenizer must be used for training and for running a model, or the numbers would mean something different.

## Token (text token, word piece)
A token is a small piece of text that a language model reads or writes, such as a whole word, part of a word or a punctuation mark. A word can be one token or several, so a model's limits are counted in tokens rather than words.

## Context window (context length)
The context window is the maximum number of tokens a language model can take into account at one time, including the prompt and its own reply. Text beyond this limit is not seen by the model, so long documents must be shortened or split.

## Large language model (LLM, language model)
A large language model is a neural network with very many parameters, trained on a huge amount of text to predict the next token. By repeating that prediction, it can write answers, summarise text and hold a conversation.

## Pre-training (base model training)
Pre-training is the first large training stage, in which a model learns general patterns from a broad dataset, such as predicting the next word in text. The result is a base model, which later training can adapt to specific tasks.

## Fine-tuning (fine tune model)
Fine-tuning continues training an already pre-trained model on a smaller, specific dataset, such as examples of helpful support replies. It changes the model's behaviour for that task while keeping much of what it learned earlier.

## LoRA (low-rank adapter, LoRA fine-tuning)
LoRA is a fine-tuning method that freezes the original model weights and trains small added matrices, called low-rank adapters, instead. The adapter is much smaller than the full model, so several task versions can share one base model.

## Adapter (model adapter, LoRA adapter)
An adapter is a small set of extra trained weights added to a frozen base model to change its behaviour for one task. Adapters are cheap to store and to swap, so one base model can serve many tasks.

## Quantisation (4-bit, model compression)
Quantisation stores a model's weights with fewer bits, which makes the file smaller and needs less memory. A 4-bit model stores each weight in 4 bits, so it has only 16 possible values per weight, instead of the 16 or 32 bits of full-precision storage, and this costs some quality.

## Distillation (knowledge distillation, teacher student)
Distillation trains a smaller student model to copy the behaviour of a larger teacher model, using the teacher's outputs as training targets. The student is cheaper to run and can keep much of the teacher's skill.

## Parameter count (model size, number of parameters)
The parameter count is the total number of learned numbers, the weights and biases, in a model. A model with more parameters usually needs more memory to load and more computing to run.

## Prompt (input prompt)
A prompt is the text or input you give a model to get a response, such as a question or an instruction with a few examples. The wording of a prompt strongly affects the answer a language model gives.

## System prompt (system message, instructions)
A system prompt is a set of instructions given to a model before the user's messages, setting its role, tone and rules. The user usually does not see it, and it shapes every reply in the conversation.

## Temperature (sampling temperature)
Temperature controls how random a language model's choice of the next token is. A low value makes the model pick the most likely tokens, giving steadier answers, while a high value makes the output more varied and more likely to contain mistakes.

## Top-p (nucleus sampling)
Top-p sampling picks the next token only from the smallest group of most likely tokens whose probabilities add up to at least p. With p set to 0.9, for example, the model chooses among the tokens that together cover 90 percent of its probability.

## Top-k (top-k sampling)
Top-k sampling picks the next token only from the k most likely tokens and ignores the rest. With k set to 1 it always takes the single most likely token, and a larger k allows more variety.

## Greedy decoding (greedy search)
Greedy decoding picks the single most likely next token at every step. For the same input it usually gives the same answer, but it can miss a sentence that would have been better overall.

## Beam search (beam decoding)
Beam search keeps several of the most promising partial outputs at each step, instead of only one, and returns the best complete one. It often gives better results than greedy decoding, for example in translation, but it needs more computing.

## Hallucination (AI making things up)
A hallucination is an answer from a language model that sounds confident but is false or made up, such as an invented book title or a fake citation. Models produce them because they generate likely-sounding text, not because they check facts against a source.

## Retrieval-augmented generation (RAG, document search with AI)
Retrieval-augmented generation first searches a collection of documents for passages related to a question, then gives those passages to the language model together with the question. This lets the model answer from up-to-date or private material that it did not learn during training.

## Vector database (vector search, embeddings store)
A vector database stores embeddings and quickly finds the stored items closest to a query embedding. It is the search part of a system that looks up passages by meaning, and for a small collection a plain file or table can do the same job.

## Cosine similarity (vector similarity)
Cosine similarity measures how closely two vectors point in the same direction, giving a value from -1 for opposite directions to 1 for identical directions. It is widely used to compare embeddings, because it looks only at direction and ignores the length of the vectors.

## KV cache (key-value cache, inference speed)
The KV cache stores the attention keys and values computed for earlier tokens, so a language model does not recompute them for every new token. It makes generation faster, but the memory it uses grows with the length of the conversation.

## Tool calling (function calling)
Tool calling lets a language model ask the program around it to run a tool, such as a calculator, a search or a file reader, and then use the result in its answer. The model writes a structured request with a tool name and arguments, and the program does the actual work.

## Agent (AI agent, autonomous agent)
An AI agent is a language model placed in a loop, where it plans steps, calls tools, reads the results and decides the next step until a task is done. The loop around the model is what lets it act, not the model alone.

## Chain of thought (step-by-step reasoning)
Chain of thought is a prompting technique in which the model writes out its intermediate reasoning steps before giving a final answer. On multi-step problems, such as word problems in arithmetic, showing the steps often leads to more accurate answers.

## Few-shot and zero-shot prompting (examples in prompt)
Few-shot prompting puts a few worked examples of the task inside the prompt, so the model copies the pattern for a new input, with no change to its weights. Zero-shot prompting gives only the instruction and no examples, which works well for common tasks but less well for unusual ones.

## Classification (classifier, categories)
Classification is a supervised task in which a model assigns each input to one category from a fixed set, such as spam or not spam. Its output is a category, not a number.

## Regression (linear regression, predicting numbers)
Regression is a supervised task in which a model predicts a number on a continuous scale, such as the price of a house from its size. Linear regression is the simplest method: it fits a straight line, or a flat plane for many inputs, by weighting each input and adding the results.

## Clustering (grouping data)
Clustering is an unsupervised task that groups items so that items in the same group are more alike than items in different groups. It needs no labels, and the groups it finds may not match any categories people had in mind.

## K-means (k-means clustering)
K-means is a clustering method that picks k cluster centres, assigns each point to its nearest centre, moves each centre to the average of its points, and repeats until the assignments stop changing. The number k must be chosen in advance.

## Decision tree (tree classifier)
A decision tree makes a prediction by asking a series of yes-or-no questions about the input features, each question splitting the data along one branch. It is easy to read and explain, but a single deep tree tends to overfit.

## Random forest (ensemble of trees)
A random forest is an ensemble of many decision trees, each trained on a random sample of the data and features, and their votes or averages form the final prediction. Averaging many varied trees usually generalises better than any one tree.

## Gradient boosting (boosted trees)
Gradient boosting builds many small decision trees one after another, where each new tree is trained to correct the errors left by the trees before it. It is a widely used method for tabular data such as spreadsheets.

## Logistic regression (classification method)
Logistic regression is a classification method that computes a weighted sum of the inputs and passes it through a sigmoid to get a probability between 0 and 1. Despite its name, it is used for classification, not for predicting numbers.

## Support vector machine (SVM, maximum margin)
A support vector machine is a classifier that finds the boundary between classes with the widest possible margin, so the boundary depends only on the training points closest to it. With kernel functions it can also draw curved boundaries.

## K-nearest neighbours (KNN, nearest neighbour)
K-nearest neighbours predicts the answer for a new item from the k training examples closest to it, using a majority vote for categories or an average for numbers. It has no real training step, but predictions get slow on very large datasets.

## Naive Bayes (probability classifier)
Naive Bayes is a classifier that estimates the probability of each class from word or feature counts, using a simple probability rule. It assumes the features do not influence each other, which is rarely exactly true, yet it works well for basic text tasks such as spam filtering.

## Precision (positive predictive value)
Precision is the share of the items a model labelled positive that really are positive. If a spam filter flags 100 emails and 90 of them are really spam, its precision is 90 percent.

## Recall (sensitivity)
Recall is the share of all the real positive items that the model found. If there are 100 spam emails and the filter catches 80 of them, its recall is 80 percent.

## F1 score (balance of precision and recall)
The F1 score combines precision and recall into one number, using their harmonic mean, so it is high only when both are high. It is useful when the classes are uneven and accuracy alone would mislead.

## Accuracy (share correct)
Accuracy is the share of all predictions that are correct. It is easy to read, but it misleads when one class is rare, because a model that always predicts the common class can score high while finding no rare cases.

## Confusion matrix (true and false positives)
A confusion matrix is a grid that counts predictions against the true labels, separating true positives, false positives, true negatives and false negatives. It shows which kinds of mistake a classifier makes, which one accuracy number hides.

## ROC curve (receiver operating characteristic)
A ROC curve plots the true positive rate against the false positive rate as the decision threshold of a classifier changes. A curve that bows toward the top left shows a better classifier, and the area under it sums up performance in one number.

## Cross-validation (k-fold validation)
Cross-validation tests a model on several different splits of the data: it trains on part of the data, tests on the rest, rotates the parts, and averages the scores. This gives a more reliable estimate than one split, especially when the dataset is small.

## Bias and variance (underfitting and overfitting trade-off)
Bias is error from a model that is too simple to capture the pattern, and variance is error from a model that reacts too strongly to the particular training data. Reducing one often increases the other, so the goal is a balance between them.

## Feature (input variable, attribute)
A feature is one measurable property of an example that is fed into a model, such as the size of a house or the number of words in an email. The choice and preparation of features affects results a great deal.

## Label (target, ground truth)
A label is the correct answer attached to a training example, such as the word spam or the price of a house. A model learns by comparing its predictions with the labels.

## Normalisation (feature scaling)
In data preparation, normalisation rescales numeric features to a common range, such as 0 to 1, so that a feature with large values does not dominate the others. It helps distance-based methods and gradient descent train more smoothly.

## One-hot encoding (categorical encoding)
One-hot encoding turns a category into a vector with a 1 at the position of that category and 0 everywhere else. With three colours, for example, red becomes `[1, 0, 0]`, green becomes `[0, 1, 0]` and blue becomes `[0, 0, 1]`.

## Dimensionality reduction (fewer features)
Dimensionality reduction turns data with many features into data with fewer features while keeping as much of the useful structure as possible. The smaller data is faster to process, easier to plot, and often less noisy.

## Principal component analysis (PCA)
Principal component analysis is a dimensionality reduction method that finds the directions in which the data varies most and keeps the top ones as new features. Each new feature is a combination of the original features, and the first ones capture the most variation.

## GAN (generative adversarial network)
A GAN is two neural networks trained against each other: a generator makes fake examples and a discriminator tries to tell them from real ones. As the discriminator improves, the generator learns to produce more realistic outputs, such as images.

## Diffusion model (image generation diffusion)
A diffusion model learns to reverse a process that gradually adds random noise to data. To create something new, it starts from pure noise and removes the noise step by step until a clean image or other output appears.

## Autoencoder (encoder decoder compression)
An autoencoder is a neural network trained to copy its input through a narrow middle layer, so the middle layer learns a compact code for the data. Autoencoders are used for compression, noise removal and spotting unusual data.

## Computer vision (image recognition)
Computer vision is the field of AI that lets computers get useful information from images and video, such as recognising objects, reading text or finding faces. It is used in medical image analysis, quality checks in factories and self-driving cars.

## Speech recognition (speech to text)
Speech recognition is the task of converting spoken audio into written text. Modern systems are mostly neural networks trained on large collections of recorded speech paired with transcripts.

## Natural language processing (NLP, text understanding)
Natural language processing is the field of AI that works with human language, covering tasks such as translation, summarising, detecting sentiment and answering questions. Large language models are now a main approach to it.

## GPU (graphics processor, why it helps AI)
A GPU, or graphics processing unit, is a chip with many simple cores that perform the same arithmetic on many numbers at once. Neural networks are mostly large matrix multiplications, so a GPU runs them much faster than a CPU, which has fewer cores.

## GGUF file (model file format)
GGUF is a single-file format for storing a language model's weights, tokenizer and settings, designed so that local programs can load it quickly. File names often include a level such as Q4, which usually means the weights are stored in about 4 bits.

## ONNX (model exchange format)
ONNX, short for Open Neural Network Exchange, is an open file format for storing a trained neural network so that it can run in different software and on different hardware. A model saved in it can be run without the code that originally trained it.

## Perplexity (language model score)
Perplexity measures how well a language model predicts a piece of text, and lower is better. A perplexity of 10 means the model is about as uncertain as if it had to choose evenly among 10 possible next tokens.

## Benchmark (model test, scores)
A benchmark is a fixed test with set tasks and a set scoring method, used to compare models on the same measure. A high score shows strength on that test only, not necessarily on every real task.

## AI alignment (AI safety, goals)
AI alignment is the work of making an AI system pursue the goals and follow the values that its designers and users actually intend.

## Turing test (machine intelligence test)
The Turing test is a way to judge whether a machine can imitate human conversation well enough that a judge cannot reliably tell it apart from a person. The judge chats by text with a hidden person and a hidden machine, then guesses which is which.
