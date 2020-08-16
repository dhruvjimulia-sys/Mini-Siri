#@title Run this code to get started
%tensorflow_version 2.x
%pip install -q transformers

import tensorflow as tf
from urllib.request import urlretrieve
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from transformers import BertTokenizer
from transformers import TFBertModel
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import SparseCategoricalAccuracy

model_name = "bert-base-cased"
tokenizer = BertTokenizer.from_pretrained(model_name)

SNIPS_DATA_BASE_URL = (
    "https://github.com/ogrisel/slot_filling_and_intent_detection_of_SLU/blob/"
    "master/data/snips/"
)
for filename in ["train", "valid", "test", "vocab.intent", "vocab.slot"]:
    path = Path(filename)
    if not path.exists():
        print(f"Downloading {filename}...")
        urlretrieve(SNIPS_DATA_BASE_URL + filename + "?raw=true", path)


def parse_line(line):
    data, intent_label = line.split(" <=> ")
    items = data.split()
    words = [item.rsplit(":", 1)[0]for item in items]
    word_labels = [item.rsplit(":", 1)[1]for item in items]
    return {
        "intent_label": intent_label, 
        "words": " ".join(words),
        "word_labels": " ".join(word_labels),
        "length": len(words),
    }

def encode_dataset(text_sequences):
    # Create token_ids array (initialized to all zeros), where 
    # rows are a sequence and columns are encoding ids
    # of each token in given sequence.
    token_ids = np.zeros(shape=(len(text_sequences), max_token_len),
                         dtype=np.int32)
    
    for i, text_sequence in enumerate(text_sequences):
        encoded = tokenizer.encode(text_sequence)
        token_ids[i, 0:len(encoded)] = encoded

    attention_masks = (token_ids != 0).astype(np.int32)
    return {"input_ids": token_ids, "attention_masks": attention_masks}


train_lines = Path("train").read_text().strip().splitlines()
valid_lines = Path("valid").read_text().strip().splitlines()
test_lines = Path("test").read_text().strip().splitlines()

df_train = pd.DataFrame([parse_line(line) for line in train_lines])
df_valid = pd.DataFrame([parse_line(line) for line in valid_lines])
df_test = pd.DataFrame([parse_line(line) for line in test_lines])

max_token_len = 43

encoded_train = encode_dataset(df_train["words"])
encoded_valid = encode_dataset(df_valid["words"])
encoded_test = encode_dataset(df_test["words"])

intent_names = Path("vocab.intent").read_text().split()
intent_map = dict((label, idx) for idx, label in enumerate(intent_names))
intent_train = df_train["intent_label"].map(intent_map).values
intent_valid = df_valid["intent_label"].map(intent_map).values
intent_test = df_test["intent_label"].map(intent_map).values

base_bert_model = TFBertModel.from_pretrained("bert-base-cased")

# Build a map from slot name to a unique id.
slot_names = ["[PAD]"] + Path("vocab.slot").read_text().strip().splitlines()
slot_map = {}
for label in slot_names:
    slot_map[label] = len(slot_map)


# Uses the slot_map of slot name to unique id, defined above, as well
# as the BERT tokenizer to create a np array with each row corresponding
# to a given sequence, and the columns as the id of the given tokens.
def encode_token_labels(text_sequences, slot_names):
    encoded = np.zeros(shape=(len(text_sequences), max_token_len), dtype=np.int32)
    for i, (text_sequence, word_labels) in enumerate( \
            zip(text_sequences, slot_names)):
        encoded_labels = []
        for word, word_label in zip(text_sequence.split(), word_labels.split()):
            tokens = tokenizer.tokenize(word)
            encoded_labels.append(slot_map[word_label])
            expand_label = word_label.replace("B-", "I-")
            if not expand_label in slot_map:
                expand_label = word_label
            encoded_labels.extend([slot_map[expand_label]] * (len(tokens) - 1))
        encoded[i, 1:len(encoded_labels) + 1] = encoded_labels
    return encoded
    
slot_train = encode_token_labels(df_train["words"], df_train["word_labels"])
slot_test = encode_token_labels(df_test["words"], df_test["word_labels"])
slot_valid = encode_token_labels(df_valid["words"], df_valid["word_labels"])

# Define the class for the model that will create predictions
# for the overall intent of a sequence, as well as the NER token labels.
class JointIntentAndSlotFillingModel(tf.keras.Model):

    def __init__(self, intent_num_labels=None, slot_num_labels=None,
                dropout_prob=0.1):
        super().__init__(name="joint_intent_slot")

        self.bert = base_bert_model
        
        # TODO: define the dropout, intent & slot classifier layers
        self.dropout = Dropout(dropout_prob)   ### YOUR CODE HERE ###
        self.intent_classifier = Dense(intent_num_labels)   ### YOUR CODE HERE ###
        self.slot_classifier = Dense(slot_num_labels)   ### YOUR CODE HERE ###

    def call(self, inputs, **kwargs):
        # Extract features from the inputs using pre-trained BERT.
        # TODO: what does the bert model return?
        sequence_output, pooled_output = self.bert(inputs, **kwargs) ### YOUR CODE HERE ###

        # TODO: use the new layers to predict slot class (logits) for each
        # token position in input sequence (size: (batch_size, seq_len, slot_num_labels)).
        sequence_output = self.dropout(sequence_output, \
                                     training=kwargs.get("training", False))
        slot_logits = self.slot_classifier(sequence_output)
        

        # TODO: define a second classification head for the sequence-wise
        # predictions (size: (batch_size, intent_num_labels)).
        # (Hint: create pooled_output to get the intent_logits).
        # Remember that the 2nd output of the main BERT layer is size 
        # (batch_size, output_dim) & gives a "pooled" representation for 
        # full sequence from hidden state corresponding to [CLS]).
        pooled_output = self.dropout(pooled_output, \
                                     training=kwargs.get("training", False))
        intent_logits = self.intent_classifier(pooled_output)

        return slot_logits, intent_logits

# TODO: create an instantiation of this model
joint_model = JointIntentAndSlotFillingModel(len(intent_map), len(slot_map))

losses = [SparseCategoricalCrossentropy(from_logits=True),
          SparseCategoricalCrossentropy(from_logits=True)]
          
joint_model.compile(optimizer=Adam(learning_rate=3e-5, epsilon=1e-08),
                    loss=losses,
                    metrics=[SparseCategoricalAccuracy('accuracy')])
                    
 # Train the model.
history = joint_model.fit(encoded_train, (slot_train, intent_train), \
    validation_data=(encoded_valid, (slot_valid, intent_valid)), \
    epochs=1, batch_size=32)
    
def show_predictions(text, intent_names, slot_names):
    inputs = tf.constant(tokenizer.encode(text))[None, :]  # batch_size = 1
    outputs = joint_model(inputs)
    slot_logits, intent_logits = outputs  ### YOUR CODE HERE ###
    slot_ids = slot_logits.numpy().argmax(axis=-1)[0, 1:-1]
    intent_id = intent_logits.numpy().argmax(axis=-1)[0]
    print("## Intent:", intent_names[intent_id])  ### YOUR CODE HERE ###
    print("## Slots:")
    for token, slot_id in zip(tokenizer.tokenize(text), slot_ids):
        print(f"{token:>10} : {slot_names[slot_id]}")
        
 def decode_predictions(text, intent_names, slot_names,
                       intent_id, slot_ids):
    info = {"intent": intent_names[intent_id]}
    collected_slots = {}
    active_slot_words = []
    active_slot_name = None
    for word in text.split():
        tokens = tokenizer.tokenize(word)
        current_word_slot_ids = slot_ids[:len(tokens)]
        slot_ids = slot_ids[len(tokens):]
        current_word_slot_name = slot_names[current_word_slot_ids[0]]
        if current_word_slot_name == "O":
            if active_slot_name:
                collected_slots[active_slot_name] = " ".join(active_slot_words)
                active_slot_words = []
                active_slot_name = None
        else:
            # Naive BIO: handling: treat B- and I- the same...
            new_slot_name = current_word_slot_name[2:]
            if active_slot_name is None:
                active_slot_words.append(word)
                active_slot_name = new_slot_name
            elif new_slot_name == active_slot_name:
                active_slot_words.append(word)
            else:
                collected_slots[active_slot_name] = " ".join(active_slot_words)
                active_slot_words = [word]
                active_slot_name = new_slot_name
    if active_slot_name:
        collected_slots[active_slot_name] = " ".join(active_slot_words)
    info["slots"] = collected_slots
    return info
 
 def nlu(text, intent_names, slot_names):
    inputs = tf.constant(tokenizer.encode(text))[None, :]  # batch_size = 1
    outputs = joint_model(inputs)
    slot_logits, intent_logits = outputs
    slot_ids = slot_logits.numpy().argmax(axis=-1)[0, 1:-1]
    intent_id = intent_logits.numpy().argmax(axis=-1)[0]

    return decode_predictions(text, intent_names, slot_names, intent_id, slot_ids)
  
 nlu("Book a table for two at Le Ritz for Friday night", intent_names, slot_names)
 nlu("Will it snow tomorrow in Saclay", intent_names, slot_names)
 nlu("I would like to listen to Anima by Thom Yorke", intent_names, slot_names)
