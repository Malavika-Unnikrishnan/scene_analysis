from flask import Flask, request, jsonify
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import io

# Initialize Flask app
app = Flask(__name__)

# Initialize the image captioning model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
model.eval()

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

@app.route('/analyze_scene', methods=['POST'])
def analyze_scene():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    # Get the image file
    image_file = request.files['image']
    
    try:
        # Convert the image file bytes to a PIL Image
        image = Image.open(io.BytesIO(image_file.read())).convert("RGB")
        
        # Preprocess the image using the processor
        processed_image = processor(images=image, return_tensors="pt").pixel_values.to(device)

        # Generate caption
        with torch.no_grad():
            output = model.generate(processed_image)

        # Decode the generated caption
        caption = processor.decode(output[0], skip_special_tokens=True)

        return jsonify({"caption": caption}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True))
