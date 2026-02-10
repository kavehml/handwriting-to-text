import json
import os
from typing import Optional, List

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import vision
from google.oauth2 import service_account
from openai import OpenAI


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
llm_client: Optional[OpenAI] = None


def _merge_lines(lines: List[str]) -> str:
    """
    Heuristically merge short broken lines into more natural sentences
    without trying to "correct" medical wording.
    """
    if not lines:
        return ""

    merged: List[str] = []
    current = lines[0]

    for line in lines[1:]:
        if not line:
            continue

        is_short = len(line.split()) <= 3
        starts_lower = line[0].islower()
        current_ends_sentence = current.endswith((".", "!", "?"))

        # If the new line is short or clearly a continuation, join it.
        if is_short or (starts_lower and not current_ends_sentence):
            if not current.endswith((" ", "\n")):
                current += " "
            current += line
        else:
            merged.append(current)
            current = line

    merged.append(current)
    return " ".join(merged)


def clean_full_text(raw_text: str) -> str:
    """
    Basic cleanup:
    - strip whitespace
    - drop empty lines
    - merge short continuation lines
    """
    raw_text = (raw_text or "").strip()
    if not raw_text:
        return ""

    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    merged = _merge_lines(lines)
    return merged.strip()


def enhance_with_llm(text: str) -> tuple[str, bool]:
    """
    Use an LLM to lightly clean up wording and ordering without
    changing the medical meaning.
    """
    text = (text or "").strip()
    if not text:
        return "", False

    if llm_client is None:
        # If no client is configured, just return the cleaned Vision text.
        return text, False

    prompt = (
        "You receive short clinical notes that have already been OCR'd from handwriting. "
        "Fix obvious spelling mistakes and word order, but do not add or remove medical "
        "information. Keep the result concise. Here is the text:\n\n"
        f"{text}\n\n"
        "Return only the corrected note, nothing else."
    )

    try:
        response = llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a careful assistant editing brief clinical notes."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=100,
        )
        corrected = response.choices[0].message.content.strip()
        return (corrected or text), True
    except Exception:
        # If the LLM call fails for any reason, fall back gracefully.
        return text, False


@app.on_event("startup")
def startup_event() -> None:
    global vision_client, llm_client
    vision_client = create_vision_client()
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        llm_client = OpenAI(api_key=api_key)


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
        image_context = vision.ImageContext(language_hints=["en"])

        # document_text_detection generally works better for handwriting
        response = vision_client.document_text_detection(image=image, image_context=image_context)

        if response.error.message:
            raise HTTPException(status_code=500, detail=response.error.message)

        text = ""
        used_llm = False
        if response.full_text_annotation and response.full_text_annotation.text:
            text = clean_full_text(response.full_text_annotation.text)
            text, used_llm = enhance_with_llm(text)

        return {"text": text, "used_llm": used_llm}
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "10000")))

