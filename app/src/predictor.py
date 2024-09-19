import pathlib

import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from ultralytics import YOLO


class TextRecognizePipeline:
    def __init__(self, detection_model_path: str, ocr_model_dir_path: str):
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        # self.device = "cpu"
        self.ocr_processor = TrOCRProcessor.from_pretrained("raxtemur/trocr-base-ru")
        self.ocr_model = VisionEncoderDecoderModel.from_pretrained(
            ocr_model_dir_path, local_files_only=True
        ).to(self.device)
        self.detection_model = YOLO(detection_model_path).to(self.device)

        # Set special tokens used for creating the decoder_input_ids from the labels.
        self.ocr_model.config.decoder_start_token_id = (
            self.ocr_processor.tokenizer.cls_token_id
        )
        self.ocr_model.config.pad_token_id = self.ocr_processor.tokenizer.pad_token_id
        # Set Correct vocab size.
        self.ocr_model.config.vocab_size = self.ocr_model.config.decoder.vocab_size
        self.ocr_model.config.eos_token_id = self.ocr_processor.tokenizer.sep_token_id

        self.ocr_model.config.max_length = 124
        self.ocr_model.config.early_stopping = True
        self.ocr_model.config.no_repeat_ngram_size = 3
        self.ocr_model.config.length_penalty = 2.0
        self.ocr_model.config.num_beams = 1
        
    def _sort_bbox_by_center(self, bbox_list):
            # Calculate center for each bounding box and sort by center_y, then center_x
            sorted_bbox = sorted(
                bbox_list, 
                key=lambda bbox: ((bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2)
            )
            return sorted_bbox

    def detect_text_and_crop_images(self, img: Image) -> list[Image]:

        result = []
        for predict, image in zip(self.detection_model.predict([img]), [img]):
            # Get bounding boxes in xyxy format (x_min, y_min, x_max, y_max)
            bboxes = predict.boxes.xyxy.cpu().tolist()
            # Sort bounding boxes by center
            sorted_bboxes = self._sort_bbox_by_center(bboxes)
            for box in sorted_bboxes:
                # Crop the image using the bounding box coordinates
                cropped_image = image.crop(box)
                # Convert to RGB
                if cropped_image.mode != 'RGB':
                    cropped_image = cropped_image.convert('RGB')
                result.append(cropped_image)
        return result

    def get_ocr_predictions(self, img_list: list[Image]) -> list[str]:
        pixel_values = self.ocr_processor(
            img_list, return_tensors="pt"
        ).pixel_values.to(self.device)
        generated_ids = self.ocr_model.generate(pixel_values)
        generated_text = self.ocr_processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        return generated_text

    def recognize(self, img: Image) -> list[str]:
        cropped_images = self.detect_text_and_crop_images(img)
        recognized_text = self.get_ocr_predictions(cropped_images)
        return cropped_images, recognized_text
    