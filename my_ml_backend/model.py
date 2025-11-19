"""
Label Studio ML Backend for Gemini VLM Pest Detection
Wraps the existing model inference engine to work with Label Studio
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
import requests

logger = logging.getLogger(__name__)


class GeminiVLMBackend(LabelStudioMLBase):
    """
    Label Studio ML Backend for Gemini VLM pest detection model.

    This backend connects to your existing model inference engine API
    and translates between Label Studio format and your API format.
    """

    # Your inference engine API endpoint
    INFERENCE_API_URL = os.getenv(
        "INFERENCE_API_URL",
        "http://localhost:5000/inference"
    )

    def __init__(self, **kwargs):
        """Initialize the ML backend."""
        super(GeminiVLMBackend, self).__init__(**kwargs)

        # Model configuration
        self.model_id = "vlm_ss"
        self.default_crop = "maize"
        self.default_task = "pest"

        # Label mapping for pest detection
        self.label_map = {
            "fall_army_worm": "Fall Army Worm",
            "sheath_blight": "Sheath Blight",
            "brown_plant_hopper": "Brown Plant Hopper",
            "pink_boll_worm": "Pink Boll Worm",
            "white_fly": "White Fly",
            "paddy_smut": "Paddy Smut",
            "rice_leaf_roller": "Rice Leaf Roller",
            "bacterial_leaf_blight": "Bacterial Leaf Blight"
        }

        logger.info(f"Initialized GeminiVLMBackend with API: {self.INFERENCE_API_URL}")

    def _get_image_url(self, task: Dict[str, Any]) -> Optional[str]:
        """
        Extract image URL from Label Studio task.

        Args:
            task: Label Studio task data

        Returns:
            Image URL or None
        """
        # Label Studio stores image URL in task['data']
        data = task.get('data', {})

        # Try different possible keys for image URL
        image_url = data.get('image') or data.get('image_url') or data.get('url')

        if not image_url:
            logger.warning(f"No image URL found in task: {task}")

        return image_url

    def _extract_crop_from_task(self, task: Dict[str, Any]) -> str:
        """
        Extract crop type from task metadata or use default.

        Args:
            task: Label Studio task data

        Returns:
            Crop type string
        """
        data = task.get('data', {})
        meta = task.get('meta', {})

        # Try to get crop from various sources
        crop = (
            data.get('crop') or
            meta.get('crop') or
            self.default_crop
        )

        return crop

    def _call_inference_api(
        self,
        image_url: str,
        crop: str,
        user_id: str = "label_studio"
    ) -> Dict[str, Any]:
        """
        Call your existing inference API.

        Args:
            image_url: URL of the image to analyze
            crop: Crop type (maize, paddy, cotton)
            user_id: User identifier

        Returns:
            API response as dictionary
        """
        payload = {
            "model_id": self.model_id,
            "task": self.default_task,
            "crop": crop,
            "user_id": user_id,
            "image_url": image_url
        }

        logger.info(f"Calling inference API with payload: {payload}")

        try:
            response = requests.post(
                self.INFERENCE_API_URL,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=120  # 2 minutes timeout
            )
            logger.info(f"printing response: {response}")
            response.raise_for_status()
            result = response.json()
            logger.info(f"printing result: {result}")
            logger.info(f"API response received: {len(result.get('detections', []))} detections")
            return result

        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling inference API: {e}")
            raise

    def _convert_to_label_studio_format(
            self,
            api_response: Dict[str, Any],
            task: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Convert your API response to Label Studio predictions format.

        Args:
            api_response: Response from your inference API
            task: Original Label Studio task

        Returns:
            List of predictions in Label Studio format
        """
        predictions = []
        detections = api_response.get('detections', [])

        if not detections:
            logger.info("No detections found in API response")
            return predictions

        # CRITICAL: Get actual image dimensions
        # Label Studio stores these when the task is created
        data = task.get('data', {})

        # Try to get dimensions from task data
        original_width = data.get('width')
        original_height = data.get('height')

        # If dimensions not in task, try to fetch from image URL
        if not original_width or not original_height:
            logger.warning("Image dimensions not found in task data, attempting to fetch from image")
            image_url = self._get_image_url(task)
            if image_url:
                try:
                    from PIL import Image
                    import requests
                    from io import BytesIO

                    response = requests.get(image_url, timeout=10)
                    img = Image.open(BytesIO(response.content))
                    original_width, original_height = img.size
                    logger.info(f"Fetched image dimensions: {original_width}x{original_height}")
                except Exception as e:
                    logger.error(f"Failed to fetch image dimensions: {e}")
                    # Fall back to detecting from bounding boxes
                    original_width = None
                    original_height = None

        # If still no dimensions, estimate from bounding boxes
        if not original_width or not original_height:
            logger.warning("Using bounding box estimation for image dimensions")
            max_x = max(det.get('box', {}).get('x_max', 0) for det in detections)
            max_y = max(det.get('box', {}).get('y_max', 0) for det in detections)
            # Add some margin (bboxes are likely not at exact edges)
            original_width = int(max_x * 1.2)
            original_height = int(max_y * 1.2)
            logger.info(f"Estimated dimensions: {original_width}x{original_height}")

        logger.info(f"Using image dimensions: {original_width}x{original_height}")

        for detection in detections:
            box = detection.get('box', {})
            label = detection.get('label', 'unknown')
            confidence = detection.get('confidence', 0.0)

            # Get readable label name
            readable_label = self.label_map.get(
                label.lower().replace(' ', '_'),
                label
            )

            # Get pixel coordinates from your API
            x_min = box.get('x_min', 0)
            y_min = box.get('y_min', 0)
            x_max = box.get('x_max', 0)
            y_max = box.get('y_max', 0)

            logger.info(f"Converting bbox: ({x_min}, {y_min}, {x_max}, {y_max}) "
                        f"with dimensions {original_width}x{original_height}")

            # Calculate percentages for Label Studio
            # Label Studio format: x, y are top-left corner as percentage
            # width and height are also percentages
            x_percent = (x_min / original_width) * 100
            y_percent = (y_min / original_height) * 100
            width_percent = ((x_max - x_min) / original_width) * 100
            height_percent = ((y_max - y_min) / original_height) * 100

            logger.info(f"Converted to percentages: x={x_percent:.2f}%, y={y_percent:.2f}%, "
                        f"w={width_percent:.2f}%, h={height_percent:.2f}%")

            # Create Label Studio prediction format
            prediction = {
                "from_name": "label",  # Must match your Label Studio config
                "to_name": "image",  # Must match your Label Studio config
                "type": "rectanglelabels",
                "value": {
                    "x": x_percent,
                    "y": y_percent,
                    "width": width_percent,
                    "height": height_percent,
                    "rotation": 0,
                    "rectanglelabels": [readable_label]
                },
                "score": confidence
            }

            predictions.append(prediction)

        logger.info(f"Converted {len(predictions)} detections to Label Studio format")
        return predictions
    def predict(
        self,
        tasks: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> ModelResponse:
        """
        Make predictions for Label Studio tasks.

        This is the main method called by Label Studio to get predictions.

        Args:
            tasks: List of Label Studio tasks
            context: Optional context data
            **kwargs: Additional parameters

        Returns:
            ModelResponse with predictions
        """
        logger.info(f"Received {len(tasks)} tasks for prediction")

        predictions = []

        for task in tasks:
            try:
                # Extract image URL
                image_url = self._get_image_url(task)
                if not image_url:
                    logger.warning(f"Skipping task {task.get('id')}: No image URL")
                    continue

                # Extract crop type
                crop = self._extract_crop_from_task(task)

                # Call your inference API
                api_response = self._call_inference_api(
                    image_url=image_url,
                    crop=crop,
                    user_id=context.get('user', {}).get('id', 'label_studio') if context else 'label_studio'
                )
                logger.info(f"printing api_response: {api_response}")

                # Convert to Label Studio format
                task_predictions = self._convert_to_label_studio_format(
                    api_response=api_response,
                    task=task
                )
                logger.info(f"printing task predictions: {task_predictions}")

                # Add model metadata
                model_version = api_response.get('model', {}).get('id', self.model_id)

                predictions.append({
                    "result": task_predictions,
                    "score": sum(p.get('score', 0) for p in task_predictions) / len(task_predictions) if task_predictions else 0,
                    "model_version": model_version
                })
                logger.info(f"printing predictions: {predictions}")

            except Exception as e:
                logger.error(f"Error processing task {task.get('id')}: {e}")
                # Return empty prediction on error
                predictions.append({
                    "result": [],
                    "score": 0,
                    "error": str(e)
                })

        logger.info(f"Returning {len(predictions)} predictions")
        return ModelResponse(predictions=predictions)

    def fit(
        self,
        event: str,
        data: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Optional: Implement training/fine-tuning logic.

        This method is called when annotations are created or updated.
        For now, we'll just log the events.

        Args:
            event: Event type (ANNOTATION_CREATED, ANNOTATION_UPDATED, etc.)
            data: Event payload
            **kwargs: Additional parameters

        Returns:
            Dictionary with status
        """
        logger.info(f"Received fit event: {event}")
        logger.debug(f"Event data: {json.dumps(data, indent=2)}")

        # Here you could:
        # 1. Collect annotations and store them
        # 2. Trigger fine-tuning of your model
        # 3. Update model parameters
        # 4. Store feedback for model improvement

        return {
            "status": "success",
            "message": f"Processed {event} event",
            "model_version": self.model_version
        }
