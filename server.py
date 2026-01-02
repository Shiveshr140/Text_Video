from fastapi import FastAPI, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from worker import task_text_to_video, task_code_to_video, task_query_to_video, celery_app
import os
import shutil

app = FastAPI()

# Enable CORS so your Frontend can call this directly
# CRITICAL: Cloudflare tunnel requires explicit CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (Cloudflare tunnel changes the origin)
    allow_credentials=False,  # Set to False when using "*" for origins
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# This lets you access files like: http://localhost:8000/files/output_final.mp4
app.mount("/files", StaticFiles(directory="."), name="files")

# CORS helper function
def add_cors_headers(response: Response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

# Handle OPTIONS requests (preflight)
@app.options("/generate/text")
@app.options("/generate/code")
@app.options("/generate/query")
@app.options("/status/{job_id}")
def options_handler():
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "*",
        }
    )

class VideoRequest(BaseModel):
    content: str
    language: str = "english"
    renderer: str = "manim"  # "manim" or "remotion"
    code_language: str = "python"  # For code videos only

# --- ENDPOINT 1: TEXT TO VIDEO ---
@app.post("/generate/text")
def generate_text(body: VideoRequest):
    # Send to background queue with renderer choice
    task = task_text_to_video.delay(
        text=body.content, 
        language=body.language, 
        renderer=body.renderer
    )
    response = Response(
        content='{"job_id": "' + task.id + '", "status": "queued", "renderer": "' + body.renderer + '"}',
        media_type="application/json"
    )
    return add_cors_headers(response)

# --- ENDPOINT 2: CODE TO VIDEO ---
@app.post("/generate/code")
def generate_code(body: VideoRequest):
    # Send to background queue with renderer and code language
    task = task_code_to_video.delay(
        code=body.content, 
        language=body.language, 
        code_language=body.code_language, 
        renderer=body.renderer
    )
    response = Response(
        content='{"job_id": "' + task.id + '", "status": "queued", "renderer": "' + body.renderer + '"}',
        media_type="application/json"
    )
    return add_cors_headers(response)

# --- ENDPOINT 3: QUERY TO VIDEO ---
@app.post("/generate/query")
def generate_query(body: VideoRequest):
    # Query always uses Manim
    task = task_query_to_video.delay(
        query=body.content, 
        language=body.language
    )
    response = Response(
        content='{"job_id": "' + task.id + '", "status": "queued", "renderer": "manim"}',
        media_type="application/json"
    )
    return add_cors_headers(response)

# --- STATUS CHECK (POLLING) ---
@app.get("/status/{job_id}")
def get_status(job_id: str):
    task = celery_app.AsyncResult(job_id)
    
    if task.state == 'SUCCESS':
        result = task.result
        # Result is like: {'final_video': 'query_job_123_final.mp4'} or full path
        
        if result and "final_video" in result:
            video_path = result['final_video']
            
            # Handle both full paths and relative paths
            if os.path.isabs(video_path):
                # Full path (from Remotion): /home/ubuntu/manim-app/remotion-video/out/text_job_XXX.mp4
                # Extract filename and copy to main directory
                filename = os.path.basename(video_path)
                dest_path = os.path.join(os.getcwd(), filename)
                
                # Copy file if it doesn't exist in main directory
                if os.path.exists(video_path) and not os.path.exists(dest_path):
                    shutil.copy(video_path, dest_path)
                    print(f"📁 Copied {video_path} -> {dest_path}")
            else:
                # Relative path (from Manim): query_job_123_final.mp4
                filename = video_path
            
            video_url = f"/files/{filename}"
            return {
                "status": "completed", 
                "video_url": video_url
            }
        else:
            return {"status": "failed", "error": "Video generation failed (No output)"}
            
    elif task.state == 'FAILURE':
        return {"status": "failed", "error": str(task.result)}
    
    # Returns "PENDING" or "STARTED"
    return {"status": "processing"}
