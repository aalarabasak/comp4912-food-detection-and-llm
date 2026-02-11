# AI and Food Detection Backend

This repository contains the Python backend server for the Malnutrition Monitoring System. Built with FastAPI, this server acts as the intelligence layer, handling computer vision tasks and Large Language Model reasoning.

The server runs locally on an Apple M3 Silicon machine and is exposed to the internet via Cloudflare Tunnel, allowing the mobile application to send secure HTTPS requests.

### API Endpoints

This backend provides two main endpoints for the mobile application:

#### 1. /detect (Object Detection)
* **Purpose:** Detects unpackaged food items (e.g., Apple, Banana, Carrot) from images uploaded by field workers.
* **Input:** Receives an image file via a POST request.
* **Process:** Uses the Ultralytics YOLOv11m model pre-trained on the COCO dataset. It processes the image using the Pillow library to handle orientation and format before inference.
* **Output:** Returns a JSON response containing the detected food class name and confidence score.

#### 2. /advice (Clinical Decision Support)
* **Purpose:** Generates personalized nutrition advice and treatment plans for children.
* **Input:** Receives a structured JSON object (FullAdviceRequest) containing the child's age, weight, risk status, recent food intake history, and nutritional targets.
* **Process:** Uses the Google Gemini 2.5 Flash API. The system constructs a detailed prompt with the child's health data and nutritional standards to generate a clinical recommendation.
* **Output:** Returns a text-based nutrition plan and treatment advice for the Nutrition Officer.

### Tech Stack
* Language: Python
* API Framework: FastAPI (Uvicorn Server)
* Object Detection: YOLOv11 (Ultralytics)
* LLM: Google Gemini 2.5 Flash (google-generativeai)
* Tunneling: Cloudflare Tunnel
* Data Validation: Pydantic
