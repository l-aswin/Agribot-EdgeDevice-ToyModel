"""
uploader.py
-----------
Handles HTTP file uploads (raw image, bounding-box image, detections JSON)
and completion notification to the web server.
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

UPLOAD_TIMEOUT = 30  # seconds per file upload


def upload_capture(
    upload_url: str,
    raw_image_path: str,
    annotated_image_path: str,
    detections_json_path: str,
    device_id: str = "jetson-nano-01",
    step_index: int = 0,
) -> bool:
    """
    Upload raw image, annotated image, and detections JSON to the server.
    Deletes local copies on success.

    Args:
        upload_url:             Server endpoint for multipart upload.
        raw_image_path:         Path to the original camera image.
        annotated_image_path:   Path to the bounding-box overlay image.
        detections_json_path:   Path to the detections JSON file.
        device_id:              Device identifier sent as form field.
        step_index:             Sequence step number for traceability.

    Returns:
        True if upload succeeded, False otherwise.
    """
    files_to_upload = {
        "raw_image":        raw_image_path,
        "annotated_image":  annotated_image_path,
        "detections_json":  detections_json_path,
    }

    # Verify all files exist before attempting upload
    for key, path in files_to_upload.items():
        if not Path(path).exists():
            logger.error("Upload file missing [%s]: %s", key, path)
            return False

    try:
        with (
            open(raw_image_path, "rb") as raw_f,
            open(annotated_image_path, "rb") as ann_f,
            open(detections_json_path, "rb") as det_f,
        ):
            files = {
                "raw_image":       (os.path.basename(raw_image_path),       raw_f,  "image/jpeg"),
                "annotated_image": (os.path.basename(annotated_image_path), ann_f,  "image/jpeg"),
                "detections":      (os.path.basename(detections_json_path), det_f,  "application/json"),
            }
            data = {"device_id": device_id, "step_index": step_index}

            logger.info("Uploading capture (step %d) to %s ...", step_index, upload_url)
            resp = requests.post(upload_url, files=files, data=data, timeout=UPLOAD_TIMEOUT)
            resp.raise_for_status()
            logger.info("Upload successful (HTTP %d).", resp.status_code)

    except requests.RequestException as e:
        logger.error("Upload failed: %s", e)
        return False

    # Delete local copies after successful upload
    for path in files_to_upload.values():
        try:
            os.remove(path)
            logger.debug("Deleted local file: %s", path)
        except OSError as e:
            logger.warning("Could not delete %s: %s", path, e)

    return True


def send_completed(completed_url: str, device_id: str, total_distance: float) -> bool:
    """
    Notify the server that the full travel distance has been covered.

    Args:
        completed_url:   Server endpoint for completion notification.
        device_id:       Device identifier.
        total_distance:  Total travel distance in meters.

    Returns:
        True if the notification was acknowledged, False otherwise.
    """
    payload = {
        "device_id":      device_id,
        "status":         "completed",
        "total_distance": total_distance,
    }
    try:
        logger.info("Sending COMPLETED notification to %s ...", completed_url)
        resp = requests.post(completed_url, json=payload, timeout=10)
        resp.raise_for_status()
        logger.info("COMPLETED notification accepted (HTTP %d).", resp.status_code)
        return True
    except requests.RequestException as e:
        logger.error("Failed to send COMPLETED notification: %s", e)
        return False
