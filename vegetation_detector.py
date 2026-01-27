"""
Greenspace Quality Feature Pipeline - Vegetation Detection with GroundingDINO and SAM
"""
import torch
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from PIL import Image
import cv2


class VegetationDetector:
    """Detect vegetation regions using GroundingDINO."""
    
    def __init__(
        self,
        vegetation_queries: Optional[List[str]] = None,
        box_threshold: float = 0.35,
        nms_iou_threshold: float = 0.5,
        device: str = "cuda"
    ):
        """
        Args:
            vegetation_queries: Text queries for vegetation detection
            box_threshold: Confidence threshold for box predictions
            nms_iou_threshold: IoU threshold for non-max suppression
            device: Device to run model on
        """
        self.device = device
        self.box_threshold = box_threshold
        self.nms_iou_threshold = nms_iou_threshold
        
        if vegetation_queries is None:
            vegetation_queries = ["vegetation", "grass", "trees", "bushes", "plants"]
        self.vegetation_queries = vegetation_queries
        self.text_prompt = " . ".join(vegetation_queries)
        
        # Load GroundingDINO
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load GroundingDINO model."""
        # Skip heavy model loading for cloud deployment - use lightweight fallback
        print("Using lightweight color-based vegetation detection (memory optimized)")
        self._use_fallback()
    
    def _download_weights(self, weights_path: str):
        """Download GroundingDINO weights."""
        import os
        import urllib.request
        
        os.makedirs(os.path.dirname(weights_path), exist_ok=True)
        
        url = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
        print(f"Downloading from {url}...")
        
        try:
            urllib.request.urlretrieve(url, weights_path)
            print("Download complete")
        except Exception as e:
            print(f"Failed to download weights: {e}")
    
    def _use_fallback(self):
        """Use a simple fallback when GroundingDINO is unavailable."""
        print("Using fallback: full-image bounding box")
        self.model = None
        self.predict_fn = None
    
    def detect_vegetation(
        self,
        image: Image.Image
    ) -> List[Dict[str, Any]]:
        """
        Detect vegetation regions in an image.
        
        Args:
            image: PIL Image (RGB)
        
        Returns:
            List of detections, each with:
                - box: (x_min, y_min, x_max, y_max) in pixel coordinates
                - score: confidence score
                - label: detected class label
        """
        if self.model is None:
            # Fallback: Color-based detection (Green + Dried Yellow)
            # Convert to HSV (OpenCV uses BGR usually, but here input is RGB PIL)
            # So convert RGB -> BGR (for cv2 standard) -> HSV? 
            # No, cv2.cvtColor(rgb_np, cv2.COLOR_RGB2HSV) works if input is RGB
            img_np = np.array(image)
            img_hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
            
            # Green range (Hue 35-85)
            lower_green = np.array([30, 40, 40])
            upper_green = np.array([90, 255, 255])
            mask_green = cv2.inRange(img_hsv, lower_green, upper_green)
            
            # Yellow/Brown range (Hue 10-30) for dried vegetation
            lower_yellow = np.array([10, 40, 40])
            upper_yellow = np.array([30, 255, 255])
            mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
            
            # Combine
            mask = cv2.bitwise_or(mask_green, mask_yellow)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            detections = []
            img_area = image.width * image.height
            min_area = img_area * 0.01  # 1% min area to avoid noise
            
            for cnt in contours:
                if cv2.contourArea(cnt) > min_area:
                    x, y, w, h = cv2.boundingRect(cnt)
                    detections.append({
                        'box': (x, y, x+w, y+h),
                        'score': 0.85, 
                        'label': 'vegetation_fallback'
                    })
            
            # If no specific regions found but some color exists?
            # Just return detection blocks.
            return detections
        
        # Transform image to Tensor for GroundingDINO
        image_trans = self.transform(image.convert("RGB"))
        
        # Run detection
        boxes, logits, phrases = self.predict_fn(
            model=self.model,
            image=image_trans,
            caption=self.text_prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.box_threshold,
            device=self.device
        )
        
        if len(boxes) == 0:
            return []
        
        # Convert normalized boxes to pixel coordinates
        w, h = image.size
        detections = []
        
        for i, (box, score, label) in enumerate(zip(boxes, logits, phrases)):
            # box is (cx, cy, w, h) normalized -> convert to (x1, y1, x2, y2) pixels
            cx, cy, bw, bh = box.tolist()
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            
            # Clamp to image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            detections.append({
                'box': (x1, y1, x2, y2),
                'score': float(score),
                'label': label
            })
        
        # Apply NMS
        detections = self._apply_nms(detections)
        
        return detections
    
    def _apply_nms(self, detections: List[Dict]) -> List[Dict]:
        """Apply non-max suppression to remove overlapping boxes."""
        if len(detections) <= 1:
            return detections
        
        boxes = np.array([d['box'] for d in detections])
        scores = np.array([d['score'] for d in detections])
        
        # Calculate IoU matrix
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1) * (y2 - y1)
        
        # Sort by score
        order = scores.argsort()[::-1]
        keep = []
        
        while len(order) > 0:
            i = order[0]
            keep.append(i)
            
            if len(order) == 1:
                break
            
            # Calculate IoU with remaining boxes
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            
            intersection = w * h
            iou = intersection / (areas[i] + areas[order[1:]] - intersection)
            
            # Keep boxes with IoU below threshold
            mask = iou <= self.nms_iou_threshold
            order = order[1:][mask]
        
        return [detections[i] for i in keep]


class SAMSegmenter:
    """Segment vegetation regions using SAM (Segment Anything Model)."""
    
    def __init__(
        self,
        model_type: str = "vit_h",
        checkpoint: Optional[str] = None,
        device: str = "cuda"
    ):
        """
        Args:
            model_type: SAM model type ('vit_h', 'vit_l', 'vit_b')
            checkpoint: Path to SAM checkpoint (auto-download if None)
            device: Device to run model on
        """
        self.device = device
        self.model_type = model_type
        self.model = None
        self.predictor = None
        
        self._load_model(checkpoint)
    
    def _load_model(self, checkpoint: Optional[str]):
        """Load SAM model."""
        try:
            from segment_anything import sam_model_registry, SamPredictor
            import os
            
            # Determine checkpoint path
            if checkpoint is None:
                checkpoint = self._get_default_checkpoint()
            
            if checkpoint and os.path.exists(checkpoint):
                print(f"Loading SAM model: {self.model_type}...")
                sam = sam_model_registry[self.model_type](checkpoint=checkpoint)
                sam = sam.to(self.device)
                self.model = sam
                self.predictor = SamPredictor(sam)
                print("SAM loaded successfully")
            else:
                print(f"SAM checkpoint not found: {checkpoint}")
                print("SAM segmentation will be unavailable")
                
        except ImportError:
            print("segment-anything not installed. SAM segmentation unavailable.")
    
    def _get_default_checkpoint(self) -> Optional[str]:
        """Get default checkpoint path."""
        import os
        
        weights_dir = os.path.join(os.path.dirname(__file__), "weights")
        os.makedirs(weights_dir, exist_ok=True)
        
        checkpoint_names = {
            "vit_h": "sam_vit_h_4b8939.pth",
            "vit_l": "sam_vit_l_0b3195.pth",
            "vit_b": "sam_vit_b_01ec64.pth",
        }
        
        checkpoint_path = os.path.join(weights_dir, checkpoint_names.get(self.model_type, "sam_vit_h_4b8939.pth"))
        
        if not os.path.exists(checkpoint_path):
            print(f"SAM checkpoint not found at {checkpoint_path}")
            print("Please download from: https://github.com/facebookresearch/segment-anything#model-checkpoints")
            return None
        
        return checkpoint_path
    
    def is_available(self) -> bool:
        """Check if SAM is available."""
        return self.model is not None
    
    def segment_boxes(
        self,
        image: Image.Image,
        boxes: List[Tuple[int, int, int, int]]
    ) -> List[np.ndarray]:
        """
        Segment regions given bounding boxes.
        
        Args:
            image: PIL Image (RGB)
            boxes: List of (x_min, y_min, x_max, y_max) boxes
        
        Returns:
            List of binary masks (H, W) as numpy arrays
        """
        if not self.is_available():
            return []
        
        if len(boxes) == 0:
            return []
        
        # Convert image to numpy
        image_np = np.array(image)
        
        # Set image
        self.predictor.set_image(image_np)
        
        masks = []
        for box in boxes:
            # Convert box to numpy array
            box_np = np.array(box)
            
            # Predict mask
            mask, _, _ = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box_np[None, :],
                multimask_output=False
            )
            
            masks.append(mask[0])  # Take first mask
        
        return masks
    
    def filter_small_masks(
        self,
        masks: List[np.ndarray],
        min_area_ratio: float = 0.01
    ) -> List[np.ndarray]:
        """
        Filter out very small masks.
        
        Args:
            masks: List of binary masks
            min_area_ratio: Minimum mask area as ratio of image size
        
        Returns:
            Filtered list of masks
        """
        if len(masks) == 0:
            return []
        
        h, w = masks[0].shape
        min_area = min_area_ratio * h * w
        
        return [m for m in masks if np.sum(m) >= min_area]
