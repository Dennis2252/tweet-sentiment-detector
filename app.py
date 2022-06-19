from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('tweetpredict.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')

def main():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def home():
    tweetComment = []
    tweet = request.form['tweetComment']
    tweetComment.append(tweet)
    tweetComment = np.array(tweetComment)
    pred = model.predict(tweetComment)
    return render_template('home.html', prediction_text=pred)

if __name__ == '__main__':
    app.run(debug=True)