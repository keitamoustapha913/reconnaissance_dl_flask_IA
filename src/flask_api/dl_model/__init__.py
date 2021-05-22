
import tensorflow as tf
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import GlobalAveragePooling2D,Dense,Flatten,Dropout,BatchNormalization,Conv2D,MaxPool2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3

import os



class dl_model:
    def __init__(self, pretrained="imagenet"):
        self.pretrained = pretrained

    
    def check_training_data(self, training_data_dir = ""):
        content=sorted(os.listdir(training_data_dir))
        print(content)
        print(f" len {len(content)} ")


    
    def train(self, training_data_dir = ""):
        data_generator = ImageDataGenerator(
            samplewise_center=True, 
            samplewise_std_normalization=True,
            brightness_range=[0.8, 1.0],
            zoom_range=[1.0, 1.2],
            validation_split=0.1
        )

        train_generator = data_generator.flow_from_directory( training_data_dir, target_size=(200,200), shuffle=True, seed=13,
                                                            class_mode='categorical', batch_size=64, subset="training")

        validation_generator = data_generator.flow_from_directory( training_data_dir, target_size=(200, 200), shuffle=True, seed=13,
                                                            class_mode='categorical', batch_size=64, subset="validation")


        inception_v3_model = InceptionV3(
            input_shape = (200, 200, 3), 
            include_top = False, 
            weights = self.pretrained ,
        )

        print("\n\n")
        #print( inception_v3_model.summary() )
        print("\n\n")

        # get the 7th mixed7 layer of inception_v3
        inception_output_layer = inception_v3_model.get_layer('mixed7')
        print('Inception model output shape:', inception_output_layer.output_shape)
        inception_output = inception_v3_model.output

        from tensorflow.keras import layers
        # Defining our output layers on top of the inception
        x =  GlobalAveragePooling2D()(inception_output)
        x =  Dense(1024, activation='relu')(x)                  
        x =  Dense(51, activation='softmax')(x)           

        self.model = Model(inception_v3_model.input, x) 

        self.model.compile(
            optimizer=SGD(lr=0.0001, momentum=0.9),
            loss='categorical_crossentropy',
            metrics=['acc']
        )
        for layer in self.model.layers[:249]:
            layer.trainable = False
        for layer in self.model.layers[249:]:
            layer.trainable = True


    def model_fit(self):

        LOSS_THRESHOLD = 0.2
        ACCURACY_THRESHOLD = 0.95

        # Stop the training when the loss is less than LOSS_THRESHOLD and accuracy is greater than ACCURACY_THRESHOLD
        class ModelCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs={}):
                if logs.get('val_loss') <= LOSS_THRESHOLD and logs.get('val_acc') >= ACCURACY_THRESHOLD:
                print("\nReached", ACCURACY_THRESHOLD * 100, "accuracy, Stopping!")
                self.model.stop_training = True

        callback = ModelCallback()


        history = self.model.fit_generator(
            train_generator,
            validation_data=validation_generator,
            steps_per_epoch=200,
            validation_steps=50,
            epochs=50,
            callbacks=[callback]
        )




if __name__ == "__main__":

    print( f"list_physical_devices {tf.config.list_physical_devices('CPU') }" )
    model = dl_model()
    training_data_dir = r"C:\Users\keita\Documents\reconnaissance_dl_flask_IA\server_api\flask_api\dl_model\dataset\ASL"
    model.check_training_data( training_data_dir = training_data_dir)
    model.train( training_data_dir = training_data_dir)

