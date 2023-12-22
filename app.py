from flask import Flask, render_template, request, redirect

# import modelnya
import pandas as pd
import pickle
with open('models/minmax.pkl', 'rb') as file:
    minmax_load = pickle.load(file)
with open('models/label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender_load = pickle.load(file)
with open('models/label_encoder_result.pkl', 'rb') as file:
    label_encoder_result_load = pickle.load(file)
with open('models/clf_gini.pkl', 'rb') as file:
    clf_gini_load = pickle.load(file)
with open('models/df.pkl', 'rb') as file:
    df_load = pickle.load(file)
with open('models/normalized_data.pkl', 'rb') as file:
    normalized_data_load = pickle.load(file)


app = Flask(__name__, static_folder='images')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/data')
def data():
    #cara1
    dataframe = pd.read_csv('data/dataframe.csv')
    dataframe_subset = dataframe.head(100)
    data_html = dataframe_subset.to_html(classes='table', index=False)
    return render_template('data.html', data_html=data_html)
    #cara2
    # # Memilih 100 data pertama dari dataframe
    # df_subset = df_load.head(100)
    # # Mengubah dataframe menjadi format HTML menggunakan metode to_html()
    # data_html = df_subset.to_html(classes='table', index=False)
    # return render_template('data.html', data_html=data_html)

@app.route('/preprocessing')
def preprocessing():
    normalized_data = pd.read_csv('data/normalized_data.csv')
    normalized_data_subset = normalized_data.head(100)
    data_html = normalized_data_subset.to_html(classes='table', index=False)
    return render_template('preprocessing.html', data_html=data_html)

@app.route('/modelling')
def modelling():
    return render_template('modelling.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # inputan
    if request.method == 'POST':
        gender = request.form['gender']
        hemoglobin = float(request.form['hemoglobin'])
        mch = float(request.form['mch'])
        mchc = float(request.form['mchc'])
        mcv = float(request.form['mcv'])
        
        input_data_numerik = [[hemoglobin, mch, mchc, mcv]]
        input_data_kategorik = [gender]
        
        # Melakukan normalisasi Min-Max Scaling pada kolom numerik
        normalisasi_numerik = minmax_load.transform(input_data_numerik)

        # Melakukan normalisasi label encoder pada kolom kategorik
        normalisasi_kategorik = label_encoder_gender_load.transform(input_data_kategorik)

        # Menggabungkan kembali data yang telah dinormalisasi
        normalized_new_data = pd.DataFrame(
            {
                'Gender': normalisasi_kategorik,
                'Hemoglobin': normalisasi_numerik[:, 0],
                'MCH': normalisasi_numerik[:, 1],
                'MCHC': normalisasi_numerik[:, 2],
                'MCV': normalisasi_numerik[:, 3]
            }
        )

        # Memprediksi dari model Decision Tree
        predicted_result = clf_gini_load.predict(normalized_new_data)

        # Mengubah hasil prediksi kembali menjadi label asli menggunakan inverse_transform pada LabelEncoder
        predicted_result = label_encoder_result_load.inverse_transform(predicted_result)

        return render_template('predict.html', prediction=predicted_result[0])
    else:
        return render_template('predict.html')
    
@app.route('/about-me')
def about_me():
    return render_template('about-me.html')


if __name__ == '__main__':
    app.run(debug=True)
