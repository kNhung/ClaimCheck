import sys
# Ensure the parent directory is in the path for imports
sys.path.append("../factchecker")

import os
import asyncio
import multiprocessing
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional
import threading
import atexit

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel

from factchecker import factcheck
import json

# Global storage for running processes with metadata
running_jobs: Dict[str, Dict] = {}
# Thread lock for job management
jobs_lock = threading.Lock()

# Ensure output directory exists
OUTPUT_DIR = Path("../reports")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Real-Time Fact-Checking API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class FactCheckRequest(BaseModel):
    claim: str
    cutoff_date: Optional[str] = None


class FactCheckResponse(BaseModel):
    claim_id: str
    status: str
    message: str


class CancelRequest(BaseModel):
    claim_id: str


def run_factcheck_process(claim: str, cutoff_date: Optional[str], claim_id: str):
    """Run fact-checking in a separate process"""

    try:
        # Call the factcheck function
        factcheck(
            claim,
            cutoff_date,
            identifier=claim_id,
            # Unified env var: FACTCHECKER_MODEL_NAME (fallback to legacy FACTCHECK_MODEL_NAME)
            model_name=os.getenv("FACTCHECKER_MODEL_NAME") or os.getenv("FACTCHECK_MODEL_NAME")
        )
        
        # Ensure the completion marker is added
        report_path = OUTPUT_DIR / claim_id / "report.json"
        # with open(report_path, "a", encoding="utf-8") as f:
        #     f.write("\n<!-- DONE -->\n")
            
    except Exception as e:
        # Write error to the report file
        report_path = OUTPUT_DIR / claim_id / "report.json"
        # try:
        #     with open(report_path, "a", encoding="utf-8") as f:
        #         f.write(f"# Error\n\nAn error occurred during fact-checking: {str(e)}\n\n<!-- DONE -->\n")
        # except Exception as file_error:
        #     print(f"Error writing to report file: {file_error}")
        #     print(f"Original error: {e}")


@app.post("/factcheck", response_model=FactCheckResponse)
async def start_factcheck(request: FactCheckRequest, background_tasks: BackgroundTasks):
    """Start a new fact-checking process"""
    print(f"Received fact-check request: {request}")
    # Generate unique claim ID
    claim_id = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")


    # Validate input
    if not request.claim.strip():
        print("Empty claim received")
        raise HTTPException(status_code=400, detail="Claim cannot be empty")
    
    # Check for too many concurrent jobs (prevent resource exhaustion)
    with jobs_lock:
        active_count = sum(1 for job_info in running_jobs.values() 
                          if job_info['process'].is_alive())
        if active_count >= 5:  # Configurable limit
            raise HTTPException(status_code=429, detail="Too many concurrent fact-checking jobs")
    
    cutoff_date = request.cutoff_date
    if not cutoff_date:
        # Default to tomorrow's date if not provided
        tomorrow = datetime.now() + timedelta(days=1)
        tomorrow_date = tomorrow.strftime("%d-%m-%Y")
        cutoff_date = tomorrow_date
    else:
        # Validate cutoff date format (DD-MM-YYYY)
        valid_formats = ["%d-%m-%Y", "%m/%d/%Y", "%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%m-%d-%Y"]
        for fmt in valid_formats:
            try:
                # Try to parse the date with each format
                datetime.strptime(cutoff_date, fmt)
                cutoff_date = datetime.strptime(cutoff_date, fmt).strftime("%d-%m-%Y")
                break
            except ValueError:
                continue
        else:
            # If no format matched, raise an error
            print(f"Invalid cutoff date format: {cutoff_date}")
            raise HTTPException(status_code=400, detail="Cutoff date must be in DD-MM-YYYY format")

    # Ensure the claim ID directory exists
    claim_dir = OUTPUT_DIR / claim_id
    os.makedirs(claim_dir, exist_ok=True)

    # Start the fact-checking process
    process = multiprocessing.Process(
        target=run_factcheck_process,
        args=(request.claim, cutoff_date, claim_id)
    )
    process.start()
    
    # Store the process reference with metadata
    with jobs_lock:
        running_jobs[claim_id] = {
            'process': process,
            'started_at': datetime.now(),
            'claim': request.claim
        }
    
    # Clean up completed processes in background
    background_tasks.add_task(cleanup_completed_jobs)
    
    return FactCheckResponse(
        claim_id=claim_id,
        status="started",
        message="Fact-checking process started successfully"
    )


@app.get("/stream-report/{claim_id}")
async def stream_report(claim_id: str):
    """
    Stream the contents of a JSON report file for a given claim_id using Server-Sent Events (SSE).
    
    Args:
        claim_id (str): The unique identifier for the claim, used to locate the report file.
    
    Returns:
        StreamingResponse: A response that streams SSE messages to the client.
    
    Raises:
        HTTPException: If the claim directory does not exist (404).
    """
    # Construct the path to the report file
    report_path = OUTPUT_DIR / claim_id / "report.json"

    # Check if the claim directory exists to avoid waiting indefinitely for an invalid claim_id
    if not report_path.parent.exists():
        raise HTTPException(status_code=404, detail="Claim not found")

    async def event_generator():
        """
        Asynchronous generator that yields SSE messages containing report updates.
        
        Yields:
            str: SSE-formatted messages (e.g., 'data: {"status": "connected", ...}\n\n').
        """
        # Send an initial 'connected' message to confirm the stream has started
        yield 'data: {}\n\n'.format(json.dumps({"status": "connected", "claim_id": claim_id}))

        # Keep track of the last modification time to detect file changes
        last_mtime = None

        while True:
            if report_path.exists():
                try:
                    # Get the current modification time of the file
                    current_mtime = report_path.stat().st_mtime

                    # If the file is new or has been modified since the last check, read and send its contents
                    if last_mtime is None or current_mtime > last_mtime:
                        with open(report_path, 'r') as f:
                            report_data = json.load(f)

                        # Send the report data as an SSE message
                        yield 'data: {}\n\n'.format(json.dumps(report_data))

                        # Update the last modification time
                        last_mtime = current_mtime

                        # Check if the report indicates completion (e.g., contains a "verdict" key)
                        if "justification" in report_data and report_data["justification"]:
                            # Send a 'complete' message and stop the stream
                            yield 'data: {}\n\n'.format(json.dumps({"status": "complete"}))
                            break

                except json.JSONDecodeError:
                    # Handle invalid JSON in the report file
                    yield 'data: {}\n\n'.format(
                        json.dumps({"status": "error", "error": "Invalid JSON in report"})
                    )
                    break
                except Exception as e:
                    # Handle other unexpected errors
                    yield 'data: {}\n\n'.format(
                        json.dumps({"status": "error", "error": str(e)})
                    )
                    break
            else:
                # If the file doesn't exist yet, send a heartbeat to keep the connection alive
                yield ': heartbeat\n\n'

            # Wait for 1 second before checking the file again
            await asyncio.sleep(1)

    # Return a StreamingResponse with the event generator and SSE media type
    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/cancel")
async def cancel_factcheck(request: CancelRequest):
    """Cancel a running fact-checking process"""
    
    claim_id = request.claim_id
    
    with jobs_lock:
        if claim_id not in running_jobs:
            raise HTTPException(status_code=404, detail="Job not found or already completed")
        
        job_info = running_jobs[claim_id]
        process = job_info['process']
        
        if process.is_alive():
            # Try graceful termination first
            process.terminate()
            
            # Wait for graceful shutdown
            process.join(timeout=5)
            
            # Force kill if still alive
            if process.is_alive():
                process.kill()
                process.join(timeout=2)

        # Remove from running jobs
        del running_jobs[claim_id]
    
    return {"status": "cancelled", "message": "Fact-checking process cancelled successfully"}


@app.get("/report/{claim_id}")
async def get_full_report(claim_id: str):
    """Get the complete report content"""
    
    report_path = OUTPUT_DIR / claim_id / "report.json"
    
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    
    with open(report_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    return {"claim_id": claim_id, "content": content}


async def cleanup_completed_jobs():
    """Clean up completed processes from memory and handle timeouts"""
    completed_jobs = []
    timed_out_jobs = []
    current_time = datetime.now()
    
    with jobs_lock:
        for claim_id, job_info in running_jobs.items():
            process = job_info['process']
            started_at = job_info['started_at']
            
            # Check if process is dead
            if not process.is_alive():
                completed_jobs.append(claim_id)
            # Check for timeout (e.g., 30 minutes)
            elif current_time - started_at > timedelta(minutes=30):
                timed_out_jobs.append(claim_id)
        
        # Clean up completed jobs
        for claim_id in completed_jobs:
            del running_jobs[claim_id]
    
    # Handle timed out jobs
    for claim_id in timed_out_jobs:
        try:
            with jobs_lock:
                if claim_id in running_jobs:
                    process = running_jobs[claim_id]['process']
                    if process.is_alive():
                        process.terminate()
                        process.join(timeout=5)
                        if process.is_alive():
                            process.kill()
                    
                    # Write timeout message
                    report_path = OUTPUT_DIR / claim_id / "report.json"
                    # with open(report_path, "a", encoding="utf-8") as f:
                    #     f.write(f"\n\n**TIMEOUT** - Process exceeded 30 minutes at {datetime.now().isoformat()}\n\n<!-- DONE -->\n")
                    
                    del running_jobs[claim_id]
        except Exception as e:
            print(f"Error handling timeout for {claim_id}: {e}")


def cleanup_all_processes():
    """Clean up all running processes on shutdown"""
    with jobs_lock:
        for claim_id, job_info in running_jobs.items():
            process = job_info['process']
            if process.is_alive():
                try:
                    process.terminate()
                    process.join(timeout=2)
                    if process.is_alive():
                        process.kill()
                except Exception:
                    pass

# Register cleanup function for graceful shutdown
atexit.register(cleanup_all_processes)


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Real-Time Fact-Checking API",
        "version": "1.0.0",
        "endpoints": {
            "POST /factcheck": "Start a new fact-checking process",
            "GET /stream-report/{claim_id}": "Stream real-time report updates",
            "POST /cancel": "Cancel a running fact-checking process",
            "GET /status/{claim_id}": "Get job status",
            "GET /report/{claim_id}": "Get complete report content"
        }
    }

@app.get("/demo")
async def demo():
    """Demo endpoint to show API functionality"""
    return FileResponse("static/demo.html", media_type="text/html")

@app.get("/demo2")
async def demo():
    """Demo endpoint to show API functionality"""
    return FileResponse("static/demo2.html", media_type="text/html")

@app.get("/demo3")
async def demo():
    """Demo endpoint to show API functionality"""
    return FileResponse("static/demo3.html", media_type="text/html")

@app.get("/demo4")
async def demo():
    """Demo endpoint to show API functionality"""
    return FileResponse("static/demo4.html", media_type="text/html")

if __name__ == "__main__":
    # CRITICAL: Set multiprocessing start method to 'spawn' to avoid issues
    # This is especially important on Unix systems and when using uvicorn
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # Method already set, which is fine
        pass
    
    # For production, consider using gunicorn or multiple uvicorn workers
    # But disable reload when using multiprocessing
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000, 
        reload=False,  # CRITICAL: Never use reload=True with multiprocessing
        access_log=True
    )
