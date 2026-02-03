from flask import Flask
app = Flask(__name__)

@app.route('/')
def home():
    return '<h1>TEST</h1><p>If you see styled text, Flask works!</p>'

if __name__ == '__main__':
    app.run(port=5001)
