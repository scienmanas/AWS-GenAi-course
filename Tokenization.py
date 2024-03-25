from transformers import AutoTokenizer
from transformers import AutoModel
import matplotlib.pyplot as plt


tokenizer = AutoTokenizer.from_pretrained("gpt-2")

# Sentence is broken into tokens, riverbank may split into river and bank
tokens = tokenizer.tokenize("A young girl named Alice sits bored by a riverbank...")

# Embeddings and processing with transfer model
# Tokens are transformed into numerical vectors. Like Alice becomes: [-0.334, 1.54, 0.23. -1.876, 0.765]
model = AutoModel.from_pretrained("gpt-2")
inputs = tokenizer("A young girl named Alice sits bored by a riverbank...", return_tensors='pt')
outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state


# Visulaization of embeddings (Simplified Example)
plt.imshow(last_hidden_states.detach().numpy()[0], cmap='virdis')
plt.colorbar()
plt.show()

'''
The core of a transformer model consists of multiple layers of self-attention and feed-forward neural networks. Each layer processes the input embeddings and passes its output to the next layer.

The actual processing involves complex mathematical operations, including self-attention mechanisms that allow each token to interact with every other token in the sequence. This is where the contextual understanding of language happens.

The final output from these layers is a set of vectors representing the input tokens in context.
'''