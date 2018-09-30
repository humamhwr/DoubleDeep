from flask import Flask, render_template, request
import process_data
import pandas as pd
import numpy as np

from keras.callbacks import ModelCheckpoint

# Configure Flask app
app = Flask(__name__)
columns_edited = False
model_rendered = False
feature_names = []
output_string = ""
neurons_per_layer = []

class CustomModel:
    number_of_features = 0

custom_model = CustomModel()

@app.route("/")
@app.route("/home")
def home():
    """Display landing page"""
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict_neural_network():
    """Predict structure"""
    global form_string
    global model_rendered
    global feature_names
    global output_string
    global neurons_per_layer

    if request.method == "POST":
        try:
            print(request.form)
            dict_answers = request.form.to_dict(flat=False)
            values_array = np.array(list(dict_answers.values())).reshape(-1)
            values_float_list = [float(i) for i in values_array]    # exclue target
            keras_model = process_data.create_keras_model(neurons_per_layer, len(feature_names))
            keras_model.load_weights("weights_best.hdf5")
            prediction = keras_model.predict(np.array([values_float_list]))
            prediction_string = "<b>Prediction</b>: "+str(prediction[0][0])
            return render_template("predict_test.html", ann_output=output_string, form=form_string, prediction=prediction_string)

        except:
            # read file
            print("Reading file...")
            file = request.files["file"]

            label_col_name = request.form["label_col_name"]
            # binary = request.form["binary_classification"]
            df = pd.read_csv(file)

            # split and scale pandas csv into data and labels
            X, Y, feature_names = process_data.process_data(df, label_col_name)
            features_dict = process_data.get_features(X)
            features_values = list(features_dict.values())
            features_values = [0.0, 0.0]+features_values

            print(features_values)
            features_values = np.array([features_values])

            # predict
            prediction, rounded_prediction = process_data.suggest_architecture(features_values)

            # append neurons per layer
            neurons_per_layer = []
            counter = 0
            print(rounded_prediction)
            for num_of_neurons in rounded_prediction:
                print(counter, num_of_neurons)
                if num_of_neurons > 1 and counter < 4:
                    neurons_per_layer.append(num_of_neurons)
                elif counter == 4:
                    neurons_per_layer.append(1)
                    break
                else:
                    break
                counter += 1
            print(neurons_per_layer)

            # create Keras model
            keras_model = process_data.create_keras_model(neurons_per_layer, len(X[0]))

            # check for binary classification
            Y_int = [int(i) for i in Y]
            if all(elem in Y_int for elem in [0,1]):
                keras_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
                print("binary!")
            else:
                keras_model.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"])
                print("mse!")
            print(X.shape)
            print(len(X[0]))

            # checkpoint
            filepath="weights_best.hdf5"
            checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
            callbacks_list = [checkpoint]

            hist = keras_model.fit(X, Y, epochs=40, batch_size=16, verbose=1, validation_split=0.2, callbacks=[checkpoint])
            model_loss = min(hist.history["val_loss"])

            baseline_model = process_data.create_baseline_model()

            if all(elem in Y_int for elem in [0,1]):
                keras_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
                print("binary!")
            else:
                keras_model.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"])
                print("mse!")
            hist_baseline = keras_model.fit(X, Y, epochs=40, batch_size=16, verbose=1, validation_split=0.2)
            basineline_min_loss = min(hist_baseline.history["val_loss"])

            counter = 0
            output_string = ''
            for neuron in neurons_per_layer:
                output_string += f"<p>Neurons per layer <b>{neuron}</b></p>"
                counter += 1

            output_string += "<br>"
            output_string += f"<p>Model loss:<b>    {model_loss}</b></p>"
            output_string += f"<p>Baseline loss:<b> {basineline_min_loss}</b></p>"

            print(output_string)

            del keras_model
            model_rendered = True

            form_string = '<form method="POST" action="/predict" class="form_format">'

            for feature_name in feature_names:
                form_string += f'<input type="number" name="{feature_name}" placeholder="{feature_name}" step=any><br>'

            form_string += '<button type="submit" class="btn btn-outline-primary" onclick="displayLoading()">Submit</button></form><br>'

            return render_template("predict_test.html", ann_output=output_string, form=form_string)

    else:
        return render_template("predict.html")

if __name__ == '__main__':
    app.run(debug=True)
