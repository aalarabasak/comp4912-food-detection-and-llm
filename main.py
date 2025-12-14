

from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Depends
#fastapi-> web framework - file-> handles file uploads - uploadfile->represents an uploaded file - HTTPException: raises HTTP errors
from fastapi.responses import JSONResponse #returns JSON responses.
from ultralytics import YOLO#loads and runs YOLOv8 models
import torch #used for device detection
from PIL import Image, ImageOps
import io#handles in-memory bytes -image data
from typing import List, Dict, Optional
import uvicorn #server to run FastAPI
import google.generativeai as genai
from pydantic import BaseModel

# API Key for authentication
API_KEY = "PP4d6Kksn9HgwVoJZ8TCUuAEpYgHTtAT"

# Google Gemini API Key
GOOGLE_API_KEY = "AIzaSyCkGctBfFBXKj_0BrFzOP9DQCPoPaIaGB0"

app = FastAPI(#Creates the FastAPI app instance
    title="Food Detection API",
    description="Detect food items in images using YOLOv8",
    version="0.1.0"
)

# Global model variable (loaded once at startup)
model = None


# API Key verification dependency
async def verify_api_key(x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
    """
    Verify API key from request header.
    Raises HTTPException if API key is missing or invalid.
    """
    if x_api_key is None:
        raise HTTPException(
            status_code=401,
            detail="API key is missing. Please provide X-API-Key header."
        )
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key. Access denied."
        )
    return x_api_key


# Pydantic model for advice request
class AdviceRequest(BaseModel):
    food_name: str


@app.on_event("startup") #runs this function when the server starts
async def load_model():#Load  model on startup

    global model
    try:
        # Load the pretrained food detection model from the Food-Detection repo this model is specially trained for food detection
        model = YOLO("yolo11m.pt")#https://docs.ultralytics.com/tr/datasets/detect/coco/#dataset-yaml
        #Loads the model from best.pt and assigns it to model

        # Verify device
        device = "mps" if torch.backends.mps.is_available() else "cpu"#checks if MPS apple silicon is available otherwise uses CPU
        print(f"model loaded successfully on {device}")#debugging message
        print(f" Model classes: {list(model.names.values())}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


@app.get("/")#handles GET requests to /.
async def root():
    return {
        "status": "online",
        "message": "Food Detection API is running",
        "device": "mps" if torch.backends.mps.is_available() else "cpu"
    }#Returns a dictionary (converted to JSON) with status info.


@app.get("/health")#handles GET requests to /health
async def health():
    #Health check endpoint - returns API status and model information
    return {
        "status": "healthy",
        "message": "Food Detection API is running",
        "device": "mps" if torch.backends.mps.is_available() else "cpu",
        "model_loaded": model is not None,
        "version": "0.1.0"
    }


@app.post("/detect")#handles POST requests to /detect
async def detect_food(
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key)  # Requires API key authentication
):

    #Detect food items in an uploaded image-uploadfile
    
    #returns json with bounding boxes confidence scores and class names

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")#checks if the model is loaded
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    try:
        # Read image file
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))

        image = ImageOps.exif_transpose(image)
        

        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Run inference in apple silicon mps-runs detection on the image using the selected device
        results = model(image, device="mps" if torch.backends.mps.is_available() else "cpu", conf = 0.15)
        
        # Parse results
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get bounding box coordinates (x1, y1, x2, y2)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()        
                # Get confidence score
                confidence = float(box.conf[0].cpu().numpy())
                
                # Get class name
                class_id = int(box.cls[0].cpu().numpy())
                class_name = model.names[class_id]
                
                detections.append({"class": class_name,
                    "confidence": round(confidence, 4),
                    "bbox": {
                        "x1": round(x1, 2),
                        "y1": round(y1, 2),
                        "x2": round(x2, 2),
                        "y2": round(y2, 2)
                    }
                })#adds detection dictionary with class confidence and rounded bbox coordinates
        
        return JSONResponse(content={
            "success": True,
            "detections": detections,
            "count": len(detections)
        })#returns a jsonresponse with successdetections list and count
        
    except Exception as e:#catchs errors if any
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post("/advice")
async def get_advice(
    request: AdviceRequest,
    api_key: str = Depends(verify_api_key)
):
    
    try:
        # Configure Gemini API
        genai.configure(api_key=GOOGLE_API_KEY) 
        
        # Create model instance
        # Using 'models/' prefix for clarity, but it works without it too
        model = genai.GenerativeModel('models/gemini-2.5-flash')
        
        # Construct prompt
        prompt = f"Tell me a short, one-sentence fun fact about {request.food_name}."
        
        # Generate content
        response = model.generate_content(prompt)
        
        # Return response
        return JSONResponse(content={
            "success": True,
            "advice": response.text.strip()
        })
        
    except Exception as e:
        print(f"gemini error {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error calling Gemini API: {str(e)}"
        )


if __name__ == "__main__": #  uv run python main.py
    uvicorn.run(
        "main:app",#app instance
        host="0.0.0.0",#accepts connections from any IP
        port=8000,#listens on port 8000
        reload=True  # autoreload on code changes 
    )
