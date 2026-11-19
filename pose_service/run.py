"""Run the independent YOLO Pose service on localhost:8001."""

import uvicorn


if __name__ == "__main__":
    uvicorn.run("pose_service.main:app", host="127.0.0.1", port=8001, reload=False)

