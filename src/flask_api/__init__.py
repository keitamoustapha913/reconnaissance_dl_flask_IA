from flask import Flask,url_for , render_template ,request
from config.flask_config import Config
import os


# Create Flask application
flask_app = Flask(__name__)
flask_app.config.from_object(Config)



@flask_app.route('/')
@flask_app.route('/home')
def home():

    return render_template("index.html")



@flask_app.route("/prediction", methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        image_file = request.files["image"]

        print(image_file.filename)
        if image_file:
            image_location = os.path.join(
                flask_app.config['UPLOAD_PRED_FOLDER'],
                image_file.filename
            )

            image_file.save(image_location)
            #pred = predict(image_location, MODEL)[0]
            pred = 1
            return render_template("prediction.html", prediction=pred, image_loc=image_file.filename)
    return render_template("prediction.html", prediction=0, image_loc=None)


@flask_app.route("/training", methods=["GET"])
def training():

    return render_template("index.html")



def create_app(config_class=Config):
    
    # random database creation
    with flask_app.app_context():

        return flask_app

