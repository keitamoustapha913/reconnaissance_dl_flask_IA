from objects.Dataset import Dataset
from objects.Callback import Callback
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import SGD


class AslPredictor:
    def __init__(self, model_params, dataset_params, callback_params):
        self.model = None
        self.dataset = Dataset(dataset_params)
        self.callback = Callback(callback_params)
        self.history = None

        # MODEL CREATION PARAMS
        self.transfer_learning_model_name = model_params['transfer_learning_model_name']
        self.lr = model_params['lr']
        self.momentum = model_params['momentum']
        self.loss_function = model_params['loss_function']
        self.metrics = model_params['metrics']
        self.first_trainable_layer = model_params['first_trainable_layer']

        # TRAINING PARAMS
        self.steps_per_epoch = model_params['steps_per_epoch']
        self.validation_steps = model_params['validation_steps']
        self.epochs = model_params['epochs']

    # __________________________________________________________________________________________________________________
    def run(self):
        self.create_model()
        self.train()
        self.evaluate()

    # __________________________________________________________________________________________________________________
    def create_model(self):
        """
            create the model
        :return:
        """

        # load already trained model for transfer learning
        inception_v3_model = keras.applications.inception_v3.InceptionV3(
            input_shape=self.input_shape,
            include_top=False,
            weights=self.transfer_learning_model_name
        )

        inception_output = inception_v3_model.output

        # add custom prediction layers
        layers = GlobalAveragePooling2D()(inception_output)
        layers = Dense(1024, activation='relu')(layers)
        layers = Dense(self.num_labels, activation='softmax')(layers)

        # create model
        model = Model(inception_v3_model.input, layers)

        # compile model
        model.compile(
            optimizer=SGD(lr=self.lr, momentum=self.momentum),
            loss=self.loss_function,
            metrics=self.metrics
        )

        # set first layers to non-trainable
        for layer in model.layers[:self.first_trainable_layer]:
            layer.trainable = False
        for layer in model.layers[self.first_trainable_layer:]:
            layer.trainable = True

        self.model = model

    # __________________________________________________________________________________________________________________
    def train(self):
        """
            train the model
        :return:
        """
        self.history = self.model.fit_generator(
            self.dataset.train_generator,
            validation_data=self.dataset.validation_generator,
            steps_per_epoch=200,
            validation_steps=50,
            epochs=50,
            callbacks=[self.callback]
        )

    # __________________________________________________________________________________________________________________
    def save_model(self, model_name: str):
        """
            save model
        :param
            model_name: name of the model
        :return:
        """
        self.model.save(model_name+'.h5')

    # __________________________________________________________________________________________________________________
    def evaluate(self):
        """
            evaluate model performances
        :return:
        """
        self.plot_loss()
        self.plot_accuracy()

    # __________________________________________________________________________________________________________________
    def plot_accuracy(self):
        """
            Plot training and validation accuracy
        :return:
        """

        acc = self.history.history['acc']
        val_acc = self.history.history['val_acc']
        epochs = range(len(acc))

        plt.plot(epochs, acc, 'r', label='Training accuracy')
        plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.legend(loc=0)
        plt.figure()

        plt.show()

    # __________________________________________________________________________________________________________________
    def plot_loss(self):
        """
            Plot training and validation loss
        :return:
        """
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']
        epochs = range(len(loss))

        plt.plot(epochs, loss, 'r', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation accuracy')
        plt.legend(loc=0)
        plt.figure()

        plt.show()

    # ==================================================================================================================
    # PROPERTIES
    @property
    def input_shape(self):
        """
            input shape of the data, which is equal to the image size + (3,) because of RGB
        :return:
        """
        return self.dataset.target_size + (3,)

    @property
    def num_labels(self):
        return len(np.unique(self.dataset.train_generator.labels))
