# Face Mask Detection 

This project is a face mask detection system using the MobileNetV2 deep learning model and Flask for the web interface. It can classify whether a person is wearing a face mask or not.

## Dependencies

Make sure you have the following Python libraries installed before running the project:

- `os`
- `numpy`
- `matplotlib`
- `keras`
- `imutils`
- `scikit-learn`
- `Flask`

You can install these libraries using `pip`:

```bash
pip install os numpy matplotlib keras imutils scikit-learn Flask
````
## Running the application
````bash
git clone https://github.com/Aswin-Sk/FaceMaskDetection.git
cd face-mask-detection
python app.py
http://localhost:5000
````

The model is already present inside mask.model. If you want to retrain the model you can make nescessary modifications to main.py and run it.

## Credits
The face mask detection model is based on the MobileNetV2 architecture.
