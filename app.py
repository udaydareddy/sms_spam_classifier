from flask import Flask, render_template, request
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
import os

app = Flask(__name__)


vectorizer_path = os.path.join(os.path.dirname(__file__), "vectorizer.pkl")
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")

tfid = pickle.load(open(vectorizer_path, 'rb'))
model = pickle.load(open(model_path, 'rb'))

ps = PorterStemmer()

def transform_message(message):
    message = message.lower()
    message = nltk.word_tokenize(message)
    y = [i for i in message if i.isalnum()]
    message = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    message = [ps.stem(i) for i in message]
    return " ".join(message)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        input_sms = request.form['message']
        transformed_sms = transform_message(input_sms)
        vector_input = tfid.transform([transformed_sms])
        prediction = model.predict(vector_input)[0]
        result = "SPAM" if prediction == 1 else "NOT SPAM"
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)