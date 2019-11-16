from flask import Flask
from flask_restplus import Resource, Api, reqparse, fields
import sys
from flask_cors import CORS
import simplejson as json
from flask_restplus import fields

# Flask Setup
app = Flask(__name__)
api = Api(app, default ="Umpires and Games")
CORS(app)
app.config["RESTPLUS_MASK_SWAGGER"] = False

# Model
hello_struct = {'message': fields.String}
hello_model = api.model('Hello Object', hello_struct)

@api.route('/hello')
class Index(Resource):
    @api.response(200, 'OK', hello_model)
    def get():
        data = json.dumps({'message': '200'}, use_decimal=True)
        js = Response(data, status=200, mimetype='application/json')
        return js

