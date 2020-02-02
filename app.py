from flask import Flask ,request,render_template

import pickle
import numpy as np
app =Flask(__name__)
model = pickle.load(open('model.pk1','rb'))

@app.route('/')
def home():
    return render_template('index.html')



@app.route('/predict',methods=['POST'])
def predict():
    int_features = [int(x) for x in  request.form.values()]
    final_features =[np.array(int_features)]
    prediction = model.predict(final_features)
    output = round(prediction[0],2)
    return render_template('index.html',prediction_text='salary should be {}'.format(output))


@app.route('/postjson',methods=['POST'])
def postJsonHandler():
    print(request.is_json)
    content =request.get_json()
    lis1 =[]
    for y in content:
        lis1.append(content[y])
    print(type(content))
    print(lis1)
    final_features =[np.array(lis1)]
    prediction = model.predict(final_features)
    output = round(prediction[0],2)
    print(output)
    return 'JSON Posted'

if __name__ == '__main__':
    app.run(debug=True)