import json
import os
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import vision
from google.oauth2 import service_account


def create_vision_client() -> vision.ImageAnnotatorClient:
    """
    Create a Google Cloud Vision client using credentials provided
    via the GOOGLE_APPLICATION_CREDENTIALS_JSON environment variable.
    """
    creds_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if not creds_json:
        raise RuntimeError(
            "GOOGLE_APPLICATION_CREDENTIALS_JSON environment variable is not set. "
            "Set it to the full JSON contents of your service account key."
        )

    try:
        info = json.loads(creds_json)
    except json.JSONDecodeError as exc:
        raise RuntimeError("GOOGLE_APPLICATION_CREDENTIALS_JSON is not valid JSON") from exc

    credentials = service_account.Credentials.from_service_account_info(info)
    return vision.ImageAnnotatorClient(credentials=credentials)


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this later to your GitHub Pages / Google Sites domains
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

vision_client: Optional[vision.ImageAnnotatorClient] = None


@app.on_event("startup")
def startup_event() -> None:
    global vision_client
    vision_client = create_vision_client()


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@app.post("/ocr")
async def ocr_image(file: UploadFile = File(...)) -> dict:
    if not vision_client:
        raise HTTPException(status_code=500, detail="Vision client is not initialized")

    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file")

        image = vision.Image(content=content)

        # document_text_detection generally works better for handwriting
        response = vision_client.document_text_detection(image=image)

        if response.error.message:
            raise HTTPException(status_code=500, detail=response.error.message)

        text = ""
        if response.full_text_annotation and response.full_text_annotation.text:
            text = response.full_text_annotation.text

        return {"text": text.strip()}
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "10000")))

