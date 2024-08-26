import tensorflow as tf 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Embedding, LSTM, Dense 
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences 
import numpy as np 
import regex as re 

def file_to_sentence_list(file_path): 
    with open(file_path, 'r') as file: 
        text = file.read() 
  
    # Splitting the text into sentences using 
    # delimiters like '.', '?', and '!' 
    sentences = [sentence.strip() for sentence in re.split( 
        r'(?<=[.!?])\s+', text) if sentence.strip()] 
  
    return sentences 
  
file_path = 'data1.txt'
text_data = file_to_sentence_list(file_path) 
  
# Tokenize the text data 
tokenizer = Tokenizer() 
tokenizer.fit_on_texts(text_data) 
total_words = len(tokenizer.word_index) + 1
  
# Create input sequences 
input_sequences = [] 
for line in text_data: 
    token_list = tokenizer.texts_to_sequences([line])[0] 
    for i in range(1, len(token_list)): 
        n_gram_sequence = token_list[:i+1] 
        input_sequences.append(n_gram_sequence) 
  
# Pad sequences and split into predictors and label 
max_sequence_len = max([len(seq) for seq in input_sequences]) 
input_sequences = np.array(pad_sequences( 
    input_sequences, maxlen=max_sequence_len, padding='pre')) 
X, y = input_sequences[:, :-1], input_sequences[:, -1] 
  
# Convert target data to one-hot encoding 
y = tf.keras.utils.to_categorical(y, num_classes=total_words) 

model = 0

def train_model():
    model = Sequential() 
    model.add(Embedding(total_words, 10, 
                        input_length=max_sequence_len-1)) 
    model.add(LSTM(128)) 
    model.add(Dense(total_words, activation='softmax')) 
    model.compile(loss='categorical_crossentropy', 
                optimizer='adam', metrics=['accuracy']) 

    #model.save('model.keras')

    from matplotlib import pyplot as plt
    history = model.fit(X, y, epochs=100, verbose=1) 
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['loss'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['accuracy', 'loss'], loc='upper left')
    plt.show()
    

def load_model():
    return tf.keras.models.load_model('model.keras')


train_model()
#model = load_model()
'''
seed_text = ""

while True:
    x = input("enter the next phrase\n")
    seed_text = seed_text + " " + x
    while True: 
        token_list = tokenizer.texts_to_sequences([seed_text])[0] 
        token_list = pad_sequences( 
            [token_list], maxlen=max_sequence_len-1, padding='pre') 
        predicted_probs = model.predict(token_list) 
        predicted_word = tokenizer.index_word[np.argmax(predicted_probs)] 
        seed_text += " " + predicted_word 
        if predicted_word == "stop":
            break
    print(seed_text)
'''
'''
ans = []
q = ["1 so ", "1 i want you to move towards me stop 2 ", "1 i want you to move towards me stop 2 keep moving towards me stop 3", "i want you to move towards me stop 2 keep moving towards me stop 3 now i want you to move to your side stop 4"]

for i in range(0,len(q)):
    seed_text = q[i]
    
    while True: 
        token_list = tokenizer.texts_to_sequences([seed_text])[0] 
        token_list = pad_sequences( 
            [token_list], maxlen=max_sequence_len-1, padding='pre') 
        predicted_probs = model.predict(token_list) 
        predicted_word = tokenizer.index_word[np.argmax(predicted_probs)] 
        seed_text += " " + predicted_word 
        if predicted_word == "stop":
            break
    
    print("Next predicted words:", seed_text) 
seed_text = "1 start by moving towards me stop 2 "

for i in range(100): 
    token_list = tokenizer.texts_to_sequences([seed_text])[0] 
    token_list = pad_sequences( 
        [token_list], maxlen=max_sequence_len-1, padding='pre') 
    predicted_probs = model.predict(token_list) 
    predicted_word = tokenizer.index_word[np.argmax(predicted_probs)] 
    seed_text += " " + predicted_word 
    


print(seed_text)
'''