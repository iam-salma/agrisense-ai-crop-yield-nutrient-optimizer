import os
import io
import torch
import timm
import numpy as np
import uvicorn
import base64
from io import BytesIO
from datetime import date
from PIL import Image
from torchvision import transforms
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

app = FastAPI(title="AI - Smart Farm Analyzer")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


loaded_models = {}
def load_model(crop_type: str):
    crop_type = crop_type.lower()
    if crop_type in loaded_models:
        return loaded_models[crop_type]

    model_path = f"models/{crop_type}_model.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model found for crop: {crop_type}")

    checkpoint = torch.load(model_path, map_location=DEVICE)
    class_names = checkpoint["class_names"]
    num_classes = len(class_names)

    model = timm.create_model("efficientnet_b2", pretrained=False, num_classes=num_classes)
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.4, inplace=True),
        torch.nn.Linear(model.classifier.in_features, num_classes)
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(DEVICE).eval()

    loaded_models[crop_type] = (model, class_names)
    return model, class_names


def pil_to_base64(img: Image.Image):
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/yield", response_class=HTMLResponse)
def yield_page(request: Request):
    return templates.TemplateResponse("yield.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
async def predict_yield(
    request: Request,
    cropType: str = Form(...),
    soilType: str = Form(...),
    location: str = Form(...),
    ph: float = Form(...),
    n: float = Form(...),
    p: float = Form(...),
    k: float = Form(...),
    mg: float = Form(...),
    temp: float = Form(...),
    rainfall: float = Form(...),
    humidity: float = Form(...)
):
    try:
        yield_pred = round(4.5 + (n + p + k + mg) / 1000, 2)
        price_pred = round(1500 + yield_pred * 50, 2)

        return templates.TemplateResponse(
            "yield.html",
            {
                "request": request,
                "cropType": cropType,
                "soilType": soilType,
                "location": location,
                "yield_pred": yield_pred,
                "price_pred": price_pred,
            },
        )
    except Exception as e:
        return templates.TemplateResponse("error.html", {"request": request, "error": str(e)})


@app.get("/nutrient", response_class=HTMLResponse)
def nutrient_page(request: Request):
    return templates.TemplateResponse("nutrient.html", {"request": request})


@app.post("/analyze-nutrients", response_class=HTMLResponse)
async def analyze_nutrients(request: Request):
    try:
        form_data = await request.form()

        result = {
            "plant_name": form_data.get("plant_name"),
            "growth_stage": form_data.get("growth_stage"),
            "soil_ph": form_data.get("soil_ph"),
            "analysis_date": date.today().strftime("%d-%m-%Y"),
            "nutrients": {
                "N": [form_data.get("nutrient_n"), "250 - 350"],
                "P": [form_data.get("nutrient_p"), "50 - 70"],
                "K": [form_data.get("nutrient_k"), "250 - 350"],
                "Ca": [form_data.get("nutrient_ca"), "200 - 300"],
                "Mg": [form_data.get("nutrient_mg"), "60 - 90"],
                "S": [form_data.get("nutrient_s"), "40 - 60"]
            },
            "statuses": {
                "N": "Low", "P": "Low", "K": "Normal",
                "Ca": "Normal", "Mg": "Normal", "S": "Normal"
            },
            "deficiencies": ["Potassium", "Calcium"],
            "excesses": ["Nitrogen"],
            "fertilizers": ["Potassium chloride (60% K)", "Potassium sulfate (50% K)", "Calcium nitrate (15% N, 19% Ca)"],
            "analysis": (
                """The analysis compares the measured nutrient concentrations in your tomatoplants (vegetative stage)
                    against established optimal ranges for this speciesand growth phase."""
            ),
            "observations": (
                """Deficiencies in K, Ca may limit plant growth and productivity.
                    Potassium deficiency in tomato leads to yellowing leaf margins and reduced diseaseresistance.
                    Calcium deficiency in tomato causes blossom-end rot in fruitsand distorted new growth.
                    Excess levels of N may cause nutrient imbalances or toxicity symptoms.
                    Excess nitrogen in tomato promotes excessive vegetative growth at theexpense of flowering/fruiting.
                    Soil pH of 7.0 is optimal for tomato cultivation. Most nutrients are optimallyavailable between pH 6.0-7.0."""
            ),
            "recommendations": (
                """Apply recommended fertilizers to address nutrient deficiencies.
                    Monitor plant growth and leaf color for signs of improvement or furtherissues.
                    Consider retesting soil and plant tissue in 2-4 weeks after fertilizerapplication.
                    Maintain proper irrigation practices as water affects nutrient availability."""
            ),
            "scientific_basis": (
                """Plant nutrient requirements vary by species and growth stage. 
                    The ideal rangesused in this analysis are based on peer-reviewed research and agriculturalextension recommendations 
                    for tomato cultivation. Nutrient interactions (e.g.,N:K ratio) and environmental factors also influence plant nutrient uptake and 
                    utilization."""
            )
        }

        return templates.TemplateResponse(
            "nutrient.html", {"request": request, "result": result}
        )
    except Exception as e:
        return templates.TemplateResponse("error.html", {"request": request, "error": str(e)})


@app.get("/deficiency", response_class=HTMLResponse)
def deficiency_page(request: Request):
    return templates.TemplateResponse("deficiency.html", {"request": request})


@app.post("/analyze-image", response_class=HTMLResponse)
async def analyze_image(request: Request, file: UploadFile = File(...), crop_type: str = Form(...)):
    try:
        contents = await file.read()
        img_pil = Image.open(io.BytesIO(contents)).convert("RGB")

        model, class_names = load_model(crop_type)
        input_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]
            pred_idx = probs.argmax().item()
            pred_class = class_names[pred_idx]
            confidence = f"{probs[pred_idx].item() * 100:.2f}%"

        # --- Grad-CAM Visualization ---
        target_layers = [model.conv_head]
        cam = GradCAM(model=model, target_layers=target_layers)
        grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(pred_idx)])[0, :]
        rgb_img = np.array(img_pil.resize((IMG_SIZE, IMG_SIZE))) / 255.0
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        visualization_uint8 = (visualization * 255).astype(np.uint8)

        original_b64 = pil_to_base64(img_pil.resize((IMG_SIZE, IMG_SIZE)))
        gradcam_b64 = pil_to_base64(Image.fromarray(visualization_uint8))

        result = {
            "predictedDeficiency": pred_class,
            "confidence": confidence,
            "uploaded_image": original_b64,
            "gradcam_image": gradcam_b64,
            "fixBudget": "₹1200",
            "fixYield": "6.2 tons/ha",
            "fixPrice": "₹1780 per ton",
            "continueYield": "4.9 tons/ha",
            "continuePrice": "₹1530 per ton"
        }

        return templates.TemplateResponse(
            "deficiency.html", {"request": request, "result": result}
        )

    except Exception as e:
        return templates.TemplateResponse(
            "error.html", {"request": request, "error": str(e)}
        )


@app.get("/getmail", response_class=HTMLResponse)
async def get_mail(request: Request, email: str):
    message = f"Thank you, {email}! 🌿 You're now subscribed to AgriAI's Smart Farming Newsletter."
    
    return templates.TemplateResponse(
        "newsletter.html", {"request": request, "message": message}
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=True)

# or write "uvicorn main:app --reload" in terminal to run the app
