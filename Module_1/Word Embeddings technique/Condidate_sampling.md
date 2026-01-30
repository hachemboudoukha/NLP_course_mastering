# Candidate Sampling

## 1. Large Vocabularies
        
To obtain good word embeddings, it is usually necessary to train an embedding model on a large amount of text data. This means that the vocabulary size will likely be very large, often reaching tens of thousands of words. However, having a large vocabulary size can significantly slow down training.

Training an embedding model is equivalent to multiclass classification, where the possible classes include every single vocabulary word. This means that we would need to calculate a softmax loss across every single vocabulary word during training, which can be incredibly time consuming for large vocabularies.

In order to mitigate the costly full softmax operation, we apply something called candidate sampling. With candidate sampling, we choose a much smaller fraction of the possible classes (i.e. vocabulary words) for computing the loss. This speeds up training significantly while also maintaining performance if we use the proper candidate samplers and loss function (more on this in the next chapter).

## 2. Computing Logits

When we calculate the loss for an embedding model, we need to first compute the model's logits. We do this by setting up trainable weights and bias terms, which will be variables created by the tf.compat.v1.get_variable function.

Similar to the final layer of a multilayer perceptron, which computes the MLP's logits, the weights and bias variables for the embedding model also compute the logits, which are then converted into the loss based on the loss function.