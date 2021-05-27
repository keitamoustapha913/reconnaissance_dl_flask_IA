from flask import Flask,url_for , render_template ,request
from config.flask_config import Config
import os

from objects.AslPredictor import AslPredictor
from config.params import get_model_params, get_dataset_params, get_callback_params

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

            predictor = AslPredictor(get_model_params(), get_dataset_params(), get_callback_params())
            class_predicted , confidence = predictor.predict(  model_name = os.path.join( flask_app.config['PROJECT_ROOT_PATH']  , "saved_model", "default_asl" ) , image_path = image_location )
            
            return render_template("prediction.html", prediction= class_predicted, image_loc=image_file.filename)

    return render_template("prediction.html", prediction= 0 , image_loc=None)


@flask_app.route("/training", methods=["GET"])
def training():
    trainer = AslPredictor(get_model_params(), get_dataset_params(), get_callback_params())
    trainer.run()
    trainer.save_model( os.path.join( flask_app.config['PROJECT_ROOT_PATH']  , "saved_model", "default_asl" ) )

    return render_template("index.html")



def create_app(config_class=Config):
    
    # random database creation
    with flask_app.app_context():

        return flask_app

