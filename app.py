import os
from flask import Flask, request, render_template, jsonify
from PIL import Image
import numpy as np
from decimal import Decimal, getcontext
import tensorflow as tf

# Set precision for Decimal operations
getcontext().prec = 6

app = Flask(__name__)

# Burn classification data with descriptions and treatment recommendations
BURN_INFO = {
    'First Degree burn': {
        'description': 'First-degree burns affect only the outer layer of skin (epidermis). The skin is red, painful, and dry with no blisters.',
        'symptoms': [
            'Redness and mild swelling',
            'Pain and tenderness',
            'Dry, peeling skin after healing begins',
            'No blisters'
        ],
        'treatment': [
            'Cool the burn with cool (not cold) running water for 10-15 minutes',
            'Apply aloe vera gel or moisturizer to keep skin hydrated',
            'Take over-the-counter pain medication if needed',
            'Avoid ice, butter, or home remedies',
            'Keep the area clean and dry',
            'Apply loose, dry bandages if needed'
        ],
        'healing_time': '3-6 days',
        'when_to_seek_help': 'Seek medical attention if the burn covers a large area, shows signs of infection, or pain persists beyond 48 hours.',
        'severity': 'Minor',
        'color': '#4CAF50'  # Green for minor
    },
    'Second Degree burn': {
        'description': 'Second-degree burns affect both the outer layer and the underlying layer of skin (dermis). These burns cause pain, redness, swelling, and blistering.',
        'symptoms': [
            'Deep redness and swelling',
            'Pain and tenderness',
            'Blisters that may break open',
            'Wet, shiny appearance',
            'Possible fever if large area affected'
        ],
        'treatment': [
            'Cool the burn immediately with cool running water for 15-20 minutes',
            'Do NOT break blisters - they protect against infection',
            'Apply antibiotic ointment if recommended by healthcare provider',
            'Cover with sterile, non-adhesive bandage',
            'Take over-the-counter pain medication',
            'Keep the area elevated if possible',
            'Stay hydrated and monitor for signs of infection'
        ],
        'healing_time': '2-3 weeks',
        'when_to_seek_help': 'SEEK IMMEDIATE MEDICAL ATTENTION. Second-degree burns require professional medical care to prevent complications and scarring.',
        'severity': 'Moderate to Severe',
        'color': '#FF9800'  # Orange for moderate
    },
    'Third Degree burn': {
        'description': 'Third-degree burns destroy both layers of skin and may affect deeper tissues. The skin appears white, black, or cherry red and may be numb.',
        'symptoms': [
            'White, black, brown, or cherry red skin',
            'Leathery or waxy texture',
            'Little to no pain (nerve endings destroyed)',
            'Possible shock symptoms',
            'Swelling in surrounding areas'
        ],
        'treatment': [
            'CALL EMERGENCY SERVICES IMMEDIATELY (123)',
            'Do NOT remove clothing stuck to burn',
            'Do NOT apply water to large third-degree burns',
            'Cover with clean, dry cloth or sterile bandage',
            'Elevate burned area above heart level if possible',
            'Monitor for shock - lay person down, elevate legs',
            'Do NOT apply ice, butter, or any home remedies'
        ],
        'healing_time': 'Months, often requires surgery',
        'when_to_seek_help': 'EMERGENCY MEDICAL ATTENTION REQUIRED IMMEDIATELY. Call 911 or go to emergency room right away.',
        'severity': 'SEVERE - EMERGENCY',
        'color': '#F44336'  # Red for severe
    }
}

def preprocessing(image):
    """Preprocess image for model prediction"""
    image = Image.open(image)
    image = image.resize((224, 224))
    image_arr = np.array(image.convert('RGB'))
    image_arr = np.expand_dims(image_arr, axis=0)  # Add batch dimension
    return image_arr.astype(np.float32)  # Ensure float32 dtype for tf.lite.Interpreter

# Initialize the TensorFlow Lite Interpreter with the model
try:
    interpreter = tf.lite.Interpreter(model_path="best_model_eff.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    interpreter = None

classes = ['First Degree burn', 'Second Degree burn', 'Third Degree burn']
thresholds = [Decimal('0.9996'), Decimal('0.99'), Decimal('0.999')]

def model_predict(image_arr):
    """Make prediction using the loaded model"""
    if interpreter is None:
        raise Exception("Model not loaded properly")
    
    interpreter.set_tensor(input_details[0]['index'], image_arr)
    interpreter.invoke()
    result = interpreter.get_tensor(output_details[0]['index'])
    return result

def get_burn_details(prediction, confidence):
    """Get detailed information about the burn classification"""
    burn_info = BURN_INFO.get(prediction, {})
    
    return {
        'classification': prediction,
        'confidence': f"{float(confidence):.2%}",
        'description': burn_info.get('description', ''),
        'symptoms': burn_info.get('symptoms', []),
        'treatment': burn_info.get('treatment', []),
        'healing_time': burn_info.get('healing_time', ''),
        'when_to_seek_help': burn_info.get('when_to_seek_help', ''),
        'severity': burn_info.get('severity', ''),
        'color': burn_info.get('color', '#666666')
    }

@app.route('/')
def index():
    return render_template('index.html', appName="AI Burn Classification & Treatment Guide")

@app.route('/health')
def health_check():
    """Health check endpoint for Render"""
    return jsonify({'status': 'healthy', 'model_loaded': interpreter is not None})

@app.route('/predictApi', methods=["POST"])
def api():
    """API endpoint for burn classification with detailed information"""
    try:
        if 'fileup' not in request.files:
            return jsonify({'error': True, 'message': "Please upload an image file"})
        
        image = request.files.get('fileup')
        if not image or image.filename == '':
            return jsonify({'error': True, 'message': "No image file selected"})
        
        # Validate image format
        try:
            image_arr = preprocessing(image)
        except Exception as e:
            return jsonify({'error': True, 'message': f"Invalid image format: {str(e)}"})
        
        result = model_predict(image_arr)
        ind = np.argmax(result)
        max_prob = Decimal(str(result[0, ind]))
        threshold = thresholds[ind]
        
        if max_prob < threshold:
            return jsonify({
                'error': False,
                'message': 'No burn detected or normal skin.',
                'recommendation': 'If you believe there is a burn injury, please retake the photo with better lighting and clarity, or consult a medical professional.'
            })
        
        prediction = classes[ind]
        burn_details = get_burn_details(prediction, max_prob)
        
        # Add emergency flag for severe burns
        burn_details['emergency'] = prediction == 'Third Degree burn'
        
        return jsonify({
            'error': False,
            'result': burn_details,
            'disclaimer': 'This AI tool is for educational purposes only and should not replace professional medical advice. Always consult healthcare professionals for proper diagnosis and treatment.'
        })
        
    except Exception as e:
        return jsonify({'error': True, 'message': f'An error occurred: {str(e)}'})

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Web interface for burn classification"""
    if request.method == 'POST':
        try:
            if 'fileup' not in request.files:
                return render_template('index.html', 
                                     error='Please select an image file',
                                     appName="AI Burn Classification & Treatment Guide")
            
            image = request.files['fileup']
            if not image or image.filename == '':
                return render_template('index.html', 
                                     error='No image file selected',
                                     appName="AI Burn Classification & Treatment Guide")
            
            image_arr = preprocessing(image)
            result = model_predict(image_arr)
            ind = np.argmax(result)
            max_prob = Decimal(str(result[0, ind]))
            threshold = thresholds[ind]
            
            if max_prob < threshold:
                return render_template('index.html', 
                                     prediction='No burn detected or normal skin.',
                                     message='If you believe there is a burn injury, please retake the photo with better lighting and clarity, or consult a medical professional.',
                                     appName="AI Burn Classification & Treatment Guide")
            
            prediction = classes[ind]
            burn_details = get_burn_details(prediction, max_prob)
            
            return render_template('index.html', 
                                 burn_info=burn_details,
                                 appName="AI Burn Classification & Treatment Guide")
            
        except Exception as e:
            return render_template('index.html', 
                                 error=f'Error processing image: {str(e)}',
                                 appName="AI Burn Classification & Treatment Guide")
    
    return render_template('index.html', appName="AI Burn Classification & Treatment Guide")

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': True, 'message': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': True, 'message': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
