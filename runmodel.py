from flask import Flask, jsonify, request, send_file
import torch
import logging
from safetensors.torch import load_file
from io import BytesIO
from PIL import Image
from diffusers import StableDiffusionPipeline
from flask_cors import CORS
from my_model import YourModel

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load model weights
model_path = "model/merged_realisticVision_VectorIllustration.safetensors"
logging.info(f"Loading model from {model_path}")

# Initialize model
model = YourModel()
model_weights = load_file(model_path)
model.load_state_dict(model_weights, strict=False)
model.eval()
logging.info("Model loaded and set to evaluation mode")

# Initialize the pipeline
pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2")
pipe.to("cuda", torch.float16)  # Use FP16 for faster performance

@app.route('/predict', methods=['POST'])
def predict():
    logging.info("Received request for prediction")

    data = request.get_json()

    prompt = data.get('prompt', '')
    negative_prompt = data.get('negative_prompt', '')
    seed = data.get('seed', torch.randint(0, 2**32 - 1, (1,)).item())  # Generate a random seed if not provided
    steps = data.get('steps', 25)
    guidance_scale = data.get('guidance_scale', 7.5)
    width = data.get('width', 576)
    height = data.get('height', 576)

    logging.info(f"Using seed: {seed}")  # Log seed for debugging

    # Set seed properly
    generator = torch.Generator().manual_seed(seed)

    # Generate the image
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        generator=generator
    ).images[0]

    if image is None:
        return jsonify({'error': 'Image generation failed'}), 500

    try:
        img_io = BytesIO()
        image.save(img_io, 'PNG')
        img_io.seek(0)
        return send_file(img_io, mimetype='image/png')
    
    except Exception as e:
        logging.error(f"Error saving image: {e}")
        return jsonify({'error': 'Image processing failed'}), 500

if __name__ == '__main__':
    logging.info("Starting Flask app")
    app.run(host='0.0.0.0', port=80)