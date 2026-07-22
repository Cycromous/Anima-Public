import re
import gc
import torch
import os
from PIL import Image
from pdf2image import convert_from_path
from transformers import NougatProcessor, VisionEncoderDecoderModel

class BaseOCREngine:
    """The standard blueprint for all Anima Vision models."""
    def __enter__(self):
        """Loads the model into VRAM."""
        return self
        
    def process(self, image_path):
        """Extracts text from a single image."""
        raise NotImplementedError("Each OCR engine must implement its own process method.")
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Flushes the model from VRAM."""
        pass

class NougatOCREngine(BaseOCREngine):
    """The Nougat-specific implementation."""
    def __enter__(self):
        print("\n   -> [SYSTEM] Loading Nougat OCR into VRAM...")
        self.processor = NougatProcessor.from_pretrained(
            "facebook/nougat-small",
            use_fast=False,
            do_crop_margin=False,
            do_thumbnail=True,
            do_align_long_axis=False
        )
        self.model = VisionEncoderDecoderModel.from_pretrained(
            "facebook/nougat-small",
            torch_dtype=torch.bfloat16
        ).to("cuda").eval()
        print("   -> [SYSTEM] Nougat Online.")
        return self

    def process(self, image_path):
        print(f"      -> Scanning {image_path}...")
        try:
            image = Image.open(image_path).convert("RGB")
            
            pixel_values = self.processor(image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.model.device, dtype=torch.bfloat16)

            with torch.no_grad():
                outputs = self.model.generate(
                    pixel_values,
                    min_length=1,
                    max_new_tokens=1500,
                    num_beams=4,
                    early_stopping=True,
                    bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                )

            sequence = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
            clean_text = self.processor.post_process_generation(sequence, fix_markdown=False)

            print(f"[NOUGAT EXTRACTION]:\n{clean_text}\n")
            return clean_text.strip()
            
        except Exception as e:
            print(f"      -> [ERROR] Failed to process {image_path}: {e}")
            return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("   -> [SYSTEM] Flushing Nougat from VRAM...")
        del self.model
        del self.processor
        gc.collect()
        torch.cuda.empty_cache()
        print("   -> [SYSTEM] VRAM restored.")


def structure_math_solution(nougat_text):
    """Structures math text into training data."""
    lines = nougat_text.split("\n")
    problem = None
    steps = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if re.search(r'[∫∑\^\(]|d[xy]|lim|sin|cos|tan', line) and problem is None:
            problem = line 
            continue
        if "=" in line:
            steps.append(line)

    if not problem:
        return None

    step_text = "\n".join([f"Step {i+1}: {s}" for i, s in enumerate(steps)])

    return {
        "messages": [
            {"role": "user", "content": f"Solve step by step: {problem}"},
            {"role": "model", "content": step_text}
        ]
    }

def pdf_to_images(pdf_path, output_folder="pdf_pages"):
    os.makedirs(output_folder, exist_ok=True)
    pages = convert_from_path(pdf_path, dpi=300)
    image_paths = []

    for i, page in enumerate(pages):
        img_path = f"{output_folder}/page_{i+1}.png"
        page.save(img_path, "PNG")
        image_paths.append(img_path)

    return image_paths