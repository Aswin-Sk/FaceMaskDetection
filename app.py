from flask import Flask, request, render_template
from test import predict_mask

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        image = request.files['file']

        if image:
            # Save the uploaded image temporarily
            image_path = 'static/uploaded_image.jpg'
            image.save(image_path)

            result = predict_mask(image_path)

            return render_template('index.html', result=result, image_path=image_path)
        else:
            return 'No file uploaded'

if __name__ == '__main__':
    app.run(debug=True)
