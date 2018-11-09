from flask import Flask, jsonify, request, make_response, abort
import json
app = Flask(__name__)


@app.route('/')
def bootstrap():
    return "Start Successful"


@app.route("/feature", methods=['POST'])
def feature():
    req_parm = request.json
    print type(req_parm)
    import faceUtils
    import ImageUtils
    image_array = ImageUtils.base64_to_array(req_parm[u'hello'])
    feature = faceUtils.face_feature(image_array)
    # face_cuts = faceUtils.face_cut(image_array)
    return json.dumps(feature,ensure_ascii=False)

    # return "successful"


if __name__ == '__main__':
    app.run(host="0.0.0.0")
