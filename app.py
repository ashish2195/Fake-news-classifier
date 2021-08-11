from flask import Flask
import joblib
app = Flask(__name__)

vectorizer = joblib.load("countvectorizer.pkl")
fakeorreal_model = joblib.load("fake_or_real.pkl")

@app.route('/')
def hello_world():
    return "Hello Ashish"

@app.route('/fakeorreal', methods=['GET', 'POST'])
def fakeorreal():
    message = request.args.get("message")
    vect_message = vectorizer.transform([message])
    result = fakeorreal_model.predict(vect_message)[0]
    return message

if __name__ == '__main__':
    app.run()


