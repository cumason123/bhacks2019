import sys
sys.path.append('./src')

from flask import Flask, jsonify, request, Response, render_template
import sys
import simplejson as json
# Flask Setup
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')
    
if __name__ == '__main__':
    if len(sys.argv) > 1:
        if 'p' in sys.argv:
            print('Starting in production mode')
            app.config["DEBUG"] = False
            app.run('0.0.0.0', port=80)
    else:
        print('Starting in testing mode')
        app.config["DEBUG"] = True
        app.run('127.0.0.1', port=3000)
