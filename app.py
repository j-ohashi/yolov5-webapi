import detector
import os
import uuid

from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename

# File
upload_dir = '/tmp/yolo'
labels = ['person','bicycle','car','motorbike','aeroplane','bus','train','truck','boat','traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','sofa','pottedplant','bed','diningtable','toilet','tvmonitor','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush']

if not os.path.exists(upload_dir):
    os.makedirs(upload_dir)
    os.chmod(upload_dir, 0o777)

# Flask
app = Flask(__name__)

def is_image(filename):
    def _is_image(form, field):
        extensions = ['jpg', 'jpeg', 'png']
        if filename and \
            filename.rsplit('.',1)[1] in extensions:
                raise ValidationError()
    return _is_image

# File Upload
def upload_file(fileobj):
    # Check if the file is an image
    if fileobj and is_image(fileobj.filename):
        # Generate temporary save file name
        file_name = str(uuid.uuid4()) +secure_filename(fileobj.filename)
        file_path = os.path.join(upload_dir, file_name)

        fileobj.save(file_path)

    # Return save file path
    return file_path

# routing
@app.route('/', methods=['POST'])
def post():

    # Receive POSTed file
    filepath = None
    if len(request.files) > 0:
        # from form item '<input type="file" name="img" accept="image/png, image/jpeg">'
        recieved_file = request.files['img']

        if (recieved_file):
            filepath = upload_file(recieved_file)

    # Receive POST data
    postdata = request.form.to_dict(flat=True)

    # Process only if the file exists
    if filepath is not None:

        # Get detect type from POST data
        detect_type = None
        if 'detecttype' in postdata:
            detect_type = request.form['detecttype']

        # Default value is weights for normal bbox, if 'seg' is set, weights for segmentation are used
        weights = 'yolov5s.pt'
        if detect_type is not None and detect_type == 'seg':
            weights = 'yolov5s-seg.pt'

        # Add any other parameters you wish to set, such as thresholds
        result = detector.run(weights, filepath, labels)
        os.remove(filepath)

        # Returns detection results in json
        try:
            output = jsonify(result=result)
        except Exception as e:
            output = jsonify(result=[])
        return output

    return jsonify(result=[])

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5000,debug=True)
