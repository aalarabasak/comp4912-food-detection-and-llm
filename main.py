

from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Depends

from fastapi.responses import JSONResponse 
from ultralytics import YOLO
import torch 
from PIL import Image, ImageOps
import io
from typing import List, Dict, Optional
import uvicorn 
import google.generativeai as genai
from pydantic import BaseModel

#API Key 
API_KEY = "PP4d6Kksn9HgwVoJZ8TCUuAEpYgHTtAT"

#Google Gemini API Key
GOOGLE_API_KEY = "AIzaSyCkGctBfFBXKj_0BrFzOP9DQCPoPaIaGB0"

app = FastAPI(#creates the fastapi app instance
    title="Food Detection API",
    description="Detect food items in images using YOLOv8",
    version="0.1.0"
)


model = None


#verify api key
async def verify_api_key(x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
  
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


#data modelss to give llm prompt
class NutritionValues(BaseModel):
    calories: float
    protein: float
    fat: float
    carbs: float


class FoodItem(BaseModel):
    name: str
    values: NutritionValues

class ChildProfile(BaseModel):
    age: str          
    gender: str
    weight: float     
    risk_status: str  
    risk_reason: str 

class FullAdviceRequest(BaseModel):
    child: ChildProfile
    target_values: NutritionValues   
    consumed_values: NutritionValues  
    rutf_inventory: List[FoodItem]    
    supplements: List[FoodItem]


@app.on_event("startup") #runs this function when the server starts
async def load_model():

    global model
    try:
       
        model = YOLO("yolo11m.pt")#https://docs.ultralytics.com/tr/datasets/detect/coco/#dataset-yaml
        

        #device verify
        device = "mps" if torch.backends.mps.is_available() else "cpu"#checks MPS is available 
        print(f"model loaded successfully on {device}")
        print(f" Model classes: {list(model.names.values())}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


@app.get("/")#handles get requests 
async def root():
    return {
        "status": "online",
        "message": "Food Detection API is running",
        "device": "mps" if torch.backends.mps.is_available() else "cpu"
    }


@app.get("/health")#handles requests to /health
async def health():
    
    return {
        "status": "healthy",
        "message": "Food Detection API is running",
        "device": "mps" if torch.backends.mps.is_available() else "cpu",
        "model_loaded": model is not None,
        "version": "0.1.0"
    }


@app.post("/detect")#handlesrequests to /detect
async def detect_food(
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key) 
):


    

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")#checks if the model is loaded
    
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    try:
        #read image file
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))

        image = ImageOps.exif_transpose(image)
        

        #turn to rgb
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        #run inference 
        results = model(image, device="mps" if torch.backends.mps.is_available() else "cpu", conf = 0.15)
        
        #parse results
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
  
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()      #get bounding box   
                confidence = float(box.conf[0].cpu().numpy()) #get score
                
                
                class_id = int(box.cls[0].cpu().numpy()) #get class name
                class_name = model.names[class_id]
                
                detections.append({"class": class_name,
                    "confidence": round(confidence, 4),
                    "bbox": {
                        "x1": round(x1, 2),
                        "y1": round(y1, 2),
                        "x2": round(x2, 2),
                        "y2": round(y2, 2)
                    }
                })
        
        return JSONResponse(content={
            "success": True,
            "detections": detections,
            "count": len(detections)
        })
        
    except Exception as e:#catchs errors 
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post("/advice")
async def get_advice(
    request: FullAdviceRequest,
    api_key: str = Depends(verify_api_key)
):
    
    try:
        #connect Gemini api
        genai.configure(api_key=GOOGLE_API_KEY) 
        

        model = genai.GenerativeModel('models/gemini-2.5-flash')

        #convert values into text 
        rutf_text = "\n".join([
            f"- {item.name}: {item.values.calories}kcal, {item.values.protein}g Prot, {item.values.fat}g Fat, {item.values.carbs}g Carb" 
            for item in request.rutf_inventory
        ])
        
        supplements_text = "\n".join([
            f"- {item.name}: {item.values.calories}kcal, {item.values.protein}g Prot, {item.values.fat}g Fat, {item.values.carbs}g Carb" 
            for item in request.supplements
        ])

        #prompt
        prompt = f"""
        Role: Act as a Nutrition Assistant for field workers. 
        Task: Write a simple 4-5 sentence recommendation in English based on the data below.

        DATA:
        - Patient: {request.child.age}, {request.child.weight}kg, Risk: {request.child.risk_status} ({request.child.risk_reason})
        - Weekly Gap (Consumed vs Target): 
          Kcal: {request.consumed_values.calories:.0f}/{request.target_values.calories:.0f}
          Protein: {request.consumed_values.protein:.0f}/{request.target_values.protein:.0f}g
          Fat: {request.consumed_values.fat:.0f}/{request.target_values.fat:.0f}g
          Carbs: {request.consumed_values.carbs:.0f}/{request.target_values.carbs:.0f}g
        
        STOCK:
        {rutf_text}
        {supplements_text}

        RULES:
        1. If Risk is 'High', YOU MUST prescribe RUTF based on weight.
        2. Identify the main nutrient gap (e.g., low protein).
        3. Suggest 1 supplement from stock to help close that gap. 
        4. Keep English very simple. No jargon.

        OUTPUT FORMAT:
        **Status:** [1 sentence summary of risk and gap]
        **Action:** [RUTF dosage and duration]
        **Diet:** [Supplement advice (in supplement list fuits)]
        """
        
        #generate content
        response = model.generate_content(prompt)

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


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True  
    )