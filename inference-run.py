'''
This is a python file to run an inference over the model built
'''

import numpy as np
import json
import re


# Loading all the weights
BIAS_HIDDEN = np.load("phase-V/projects/ch-28/v2/weights/BIAS_HIDDEN.npy")
WEIGHT_HIDDEN = np.load("phase-V/projects/ch-28/v2/weights/WEIGHT_HIDDEN.npy")
WEIGHT_INPUT = np.load("phase-V/projects/ch-28/v2/weights/WEIGHT_INPUT.npy")
WEIGHT_OUTPUT = np.load("phase-V/projects/ch-28/v2/weights/WEIGHT_OUTPUT.npy")
BIAS_OUTPUT = np.load("phase-V/projects/ch-28/v2/weights/BIAS_OUTPUT.npy")
VOCABULARY = np.load("phase-V/projects/ch-28/v2/weights/VOCABULARY.npy")

# loading the token_ids
with open("phase-V/projects/ch-28/v2/weights/token_id.json") as file:
    TOKEN_ID = json.load(file)


# text normalization methods
def textNormalization(string: str) -> str:
    '''
    This function will take in a string and return a clean string 
    '''
    
    space_pattern = r"\s+"
    
    string = string.lower()
    string = string.strip()
    string = re.sub(space_pattern, " ", string)
    
    return string

def textTokenization(string: str) -> list[str]:
    '''
    This function is to do tokenization of the string
    '''
    
    return string.split()

def textVectorization(array: str, prev_vocab:dict , counter: int = 2, ) -> dict:
    '''
    this function is to do text vectorization
    0 ID is reserved for padding token
    1 ID is reserved for Unkown token
    '''
    
    i = counter
    vocab = {}
    for token in array:
        # if token not in vocab.keys() and token not in prev_vocab.keys():
        vocab[token] = i
        i += 1
    return vocab, i

# important variables
STATE_HIDDEN = []
STATE_INPUT = []

# sigmoid function
def sigmoid(x):
    return 1/ (1 + np.exp(-x))

# main rnn function
def rnn(sequence, state) -> list:
    global WEIGHT_INPUT, WEIGHT_HIDDEN, BIAS_HIDDEN, STATE_EMBEDDING
    '''
    This is going to be an iterative function which is going to create a hidden state and is going to process the incoming tokens
    
    the input dimension of the sequence must be 5x1
    
    '''
    
    # # normalizing the sequence
    sequence = textNormalization(sequence)
    sequence = textTokenization(sequence)
    
    # applying the strict limit
    sequence = sequence[:120]
    
    for word in sequence:
        # retrieving the embedding from the sequence (tokenization)
        token = TOKEN_ID.get(word, TOKEN_ID["<UNK>"])
        embedding = VOCABULARY[token].reshape(25,1)   # reshaping for compatibility
        
        # calculating hidden state
        state = sigmoid( 
                        WEIGHT_INPUT @ embedding + 
                        WEIGHT_HIDDEN @ state + 
                        BIAS_HIDDEN 
        ) # it's output dimension will be 5x1 (a column vector) or a row vector upon taking transpose
        
        # adding the values to the state
        STATE_HIDDEN.append(state)
        STATE_INPUT.append(embedding)

    return WEIGHT_INPUT, WEIGHT_HIDDEN, BIAS_HIDDEN, state

# main inference method
def inference(x: str) -> float:
    global STATE_HIDDEN, STATE_INPUT # defining the global scope
    
   
    STATE_HIDDEN = [] # intializing the hidden state
    STATE_INPUT = [] # intializing the input state
    
    # creating a 5 token sequence for the purpose of debugging
    # token = textTokenization(textNormalization(x))
    # token = token[:5]
    
    # x = " ".join(token)
    
    # rnn cell
    rnn_cell = rnn(x, np.zeros(25).reshape(25,1))
    
    output = WEIGHT_OUTPUT @ rnn_cell[3] + BIAS_OUTPUT # calculating the main output itself
    z = sigmoid(output) #applying the activation function 
    
    return z # returning the final value


X = input("Enter your Review: ")

Y = inference(X)
if Y[0][0] > .5:
    output = "Positive"
else:
    output = "Negative"

print(f"KhushiRNN: The Review is {output}")

