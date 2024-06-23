from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import base64
import os
from uuid import uuid4
import uvicorn
import test
app = FastAPI()

class ImageUpload(BaseModel):
    base64_image: str

@app.post("/upload/")
async def upload_image(image: ImageUpload):
    try:
        # Decode the base64 image
        image_data = base64.b64decode(image.base64_image)
        # Create a unique file name
        file_name = f"{uuid4().hex}.png"
        file_path = os.path.join("uploaded_images", file_name)

        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Write the image data to a file
        with open(file_path, "wb") as file:
            file.write(image_data)

        # Return the file URL
        file_url = f"http://localhost:8000/files/{file_name}"
        respond=test.generate(file_url, "can you tell me about this image?")
        return respond

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
