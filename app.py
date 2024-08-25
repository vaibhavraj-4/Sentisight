from flask import Flask, render_template, request, jsonify
from flask import url_for
from textblob import TextBlob
import nltk


nltk.download(info_or_id="vader_lexicon")
from nltk.sentiment.vader import SentimentIntensityAnalyzer

app = Flask(__name__)

FORM_TEMPLATE: str = "form.html"
ANALYSE_TEMPLATE: str = "analyzed.html"


@app.route("/", methods=["GET"])
def index():
    print(url_for("static", filename="logo.svg"))
    return render_template(FORM_TEMPLATE, blob_sentiment=0)


@app.route("/result", methods=["POST"])
def analyse():
    text = request.form.get("rawtext")
    return render_template(ANALYSE_TEMPLATE, text=text)


@app.route("/process", methods=["GET"])
def processText():
    # print("Processing...")
    text = request.args.get("text")

    sid = SentimentIntensityAnalyzer()
    process_dict: dict[str, float] = sid.polarity_scores(text)

    compound: float = process_dict["compound"]
    if compound == 0:
        sentiment_type = "Neutral"
    elif compound > 0:
        sentiment_type = "Positive"
    elif compound < 0:
        sentiment_type = "Negative"

    data = dict()
    data["sentiment_type"] = sentiment_type
    data["polarity"] = round(TextBlob(text=text).sentiment.polarity, 2)
    data["positive_pcnt"] = round(process_dict["pos"] * 100, 2)
    data["neutral_pcnt"] = round(process_dict["neu"] * 100, 2)
    data["negative_pcnt"] = round(process_dict["neg"] * 100, 2)
    data["text"] = text
    return jsonify(data)


if __name__ == "__main__":
    app.run(debug=True)
