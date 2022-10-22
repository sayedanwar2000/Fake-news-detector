import numpy as np
from flask import Flask, request, jsonify
import pickle
from keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string 
from bs4 import BeautifulSoup
from keras.preprocessing import text,sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Dense,Embedding,LSTM,Dropout
from keras.layers import Dense,Embedding,LSTM,Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle
from keras.models import load_model
from bs4 import BeautifulSoup
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import keras
from keras.preprocessing import text,sequence
from keras.models import Sequential
from keras.layers import Dense,Embedding,LSTM,Dropout
from sklearn.neural_network import MLPClassifier
from keras.models import load_model
import pickle
import tensorflow as tf
from tensorflow import keras
import time

pickle_in = open(r"C:\Users\youssef.YOUSSEF\medel_deployment\tokenizer.pkl",'rb')
tokenizer = pickle.load(pickle_in)

model = load_model('model.h5')

class news_detection :
  def __init__(self,news):
    import pandas as pd
    self.news = news 
    

    print(text)

  @staticmethod
  def remove_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

#Removal of Punctuation Marks

  @staticmethod
  def remove_punctuations(text):
    return re.sub('\[[^]]*\]', '', text)

# Removal of Special Characters
  @staticmethod
  def remove_characters(text):
    return re.sub("[^a-zA-Z]"," ",text)


#Total function
  def cleaning(self):
    self.news = self.remove_html(self.news)
    self.news = self.remove_punctuations(self.news)
    self.news = self.remove_characters(self.news)
    return self.news

  @staticmethod  
  def output_lable(n):
    if n :
        return "Real News"
    else :
        return "Fake News"
  


  def manual_testing(self):
    import pandas as pd

    testing_news = {"text":[self.cleaning()]}
    new_def_test = pd.DataFrame(testing_news)
    # new_def_test["text"] = new_def_test["text"].apply(self.cleaning) 
    new_x_test = new_def_test["text"]
    new_xv_test = tokenizer.texts_to_sequences(new_x_test)
    new_xv_test = tf.keras.preprocessing.sequence.pad_sequences(new_xv_test, maxlen=300)

    pred = model.predict(new_xv_test)
    pred = pred > 0.5
    
    return(self.output_lable(pred))


test = "MOSCOW (Reuters) Vatican Secretary of State Cardinal Pietro Parolin said on Tuesday that there was  positive momentum  behind the idea of Pope Francis visiting Russia, but suggested there was more work to be done if it were to happen.  Parolin, speaking at a joint news conference in Moscow alongside Russian Foreign Minister Sergei Lavrov, did not give any date for such a possible visit. The Eastern and Western branches of Christianity split apart in 1054. The pope, leader of the world s 1.2 billion Catholics, is seeking to improve ties, and last year in Cuba held what was the first ever meeting between a Roman Catholic pope and a Russian Orthodox patriarch.  Parolin said he had also used his talks in the Russian capital to also raise certain difficulties faced by the Catholic Church in Russia. He said that Moscow and the Vatican disagreed about the plight of Christians in certain parts of the world. He did not elaborate. Parolin, who is due later on Tuesday to meet Patriarch Kirill, the head of the Russian Orthodox Church, said he also believed Russia could play an important role when it came to helping solve a crisis in Venezuela because of its close relations with Caracas."

new_class = pickle.load(open(r'news_detection.pkl','rb' ))

obj = news_detection(test)
print(obj.cleaning())

print( obj.manual_testing() )


app = Flask(__name__)

@app.route('/',methods=['POST'])
def predict():
    test = "MOSCOW (Reuters) Vatican Secretary of State Cardinal Pietro Parolin said on Tuesday that there was  positive momentum  behind the idea of Pope Francis visiting Russia, but suggested there was more work to be done if it were to happen.  Parolin, speaking at a joint news conference in Moscow alongside Russian Foreign Minister Sergei Lavrov, did not give any date for such a possible visit. The Eastern and Western branches of Christianity split apart in 1054. The pope, leader of the world s 1.2 billion Catholics, is seeking to improve ties, and last year in Cuba held what was the first ever meeting between a Roman Catholic pope and a Russian Orthodox patriarch.  Parolin said he had also used his talks in the Russian capital to also raise certain difficulties faced by the Catholic Church in Russia. He said that Moscow and the Vatican disagreed about the plight of Christians in certain parts of the world. He did not elaborate. Parolin, who is due later on Tuesday to meet Patriarch Kirill, the head of the Russian Orthodox Church, said he also believed Russia could play an important role when it came to helping solve a crisis in Venezuela because of its close relations with Caracas."
    
    
    
    data = request.get_json(force=True)
    obj = new_class(str(data.value()))
    prediction = obj.manual_testing()
    return jsonify(prediction)



if __name__ == "__main__":
    app.run(debug=True)

#url = 'http://localhost:5000/'

#import requests
#r = requests.post(url)


