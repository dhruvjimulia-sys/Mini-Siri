import tensorflow as tf
from pathlib import Path
from transformers import BertTokenizer
import pickle

with open('minisirimodel.pickle', 'rb') as m:
    joint_model = pickle.load(m)

model_name = "bert-base-cased"
tokenizer = BertTokenizer.from_pretrained(model_name)

intent_names = Path("vocab.intent").read_text().split()
slot_names = ["[PAD]"] + Path("vocab.slot").read_text().strip().splitlines()

def show_predictions(text, intent_names, slot_names):
    inputs = tf.constant(tokenizer.encode(text))[None, :]  # batch_size = 1
    outputs = joint_model(inputs)
    slot_logits, intent_logits = outputs
    slot_ids = slot_logits.numpy().argmax(axis=-1)[0, 1:-1]
    intent_id = intent_logits.numpy().argmax(axis=-1)[0]
    print("## Intent:", intent_names[intent_id])
    print("## Slots:")
    for token, slot_id in zip(tokenizer.tokenize(text), slot_ids):
        print(f"{token:>10} : {slot_names[slot_id]}")
        
def decode_predictions(text, intent_names, slot_names, intent_id, slot_ids):
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

user_continue = "Y"
while (user_continue.upper() != "N"):
    usr_input_str = input("Type in a command for Mini-Siri: ")
    print(nlu(usr_input_str, intent_names, slot_names))
    user_continue = input("Do you want to continue asking to Mini-Siri?(Y/N): ")
print("Turned off Mini-Siri")