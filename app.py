from flask import Flask, request, render_template
from nltk.sentiment.vader import SentimentIntensityAnalyzer

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if request.method == 'POST':
        text = request.form['text']
        sid = SentimentIntensityAnalyzer()
        sentiment = sid.polarity_scores(text)
        return render_template('result.html', sentiment=sentiment, text=text)

if __name__ == '__main__':
    app.run(debug=True)
