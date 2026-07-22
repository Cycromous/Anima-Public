import os
import json
import uuid
import gc
import torch
from transformers import NougatProcessor, VisionEncoderDecoderModel
from Occipital_vision import NougatOCREngine, structure_math_solution, pdf_to_images

DATASET_FILE = "./ai_memory/training_data.jsonl"

def append_to_training_dataset(extracted_text):
    structured = structure_math_solution(extracted_text)
    if not structured:
        return
    os.makedirs(os.path.dirname(DATASET_FILE), exist_ok=True)
    with open(DATASET_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(structured) + "\n")
    print(f"[DATASET] Added example. File size: {os.path.getsize(DATASET_FILE)} bytes.")

def _load_nougat():
    """JIT loads Nougat into VRAM. Called only when needed."""
    print("\n   -> [SYSTEM] Loading Nougat OCR into VRAM...")
    processor = NougatProcessor.from_pretrained(
        "facebook/nougat-small",
        use_fast=False,
        do_crop_margin=False,
        do_thumbnail=True,
        do_align_long_axis=False
    )
    model = VisionEncoderDecoderModel.from_pretrained(
        "facebook/nougat-small",
        torch_dtype=torch.bfloat16
    ).to("cuda").eval()
    print("   -> [SYSTEM] Nougat Online.")
    return model, processor

def _flush_nougat(model, processor):
    """Flushes Nougat from VRAM immediately after use."""
    print("   -> [SYSTEM] Flushing Nougat from VRAM...")
    del model
    del processor
    gc.collect()
    torch.cuda.empty_cache()
    print("   -> [SYSTEM] VRAM restored.")

def learn_from_pdf(pdf_path, collection):
    """
    JIT loads Nougat, processes the PDF, then immediately flushes VRAM.
    Signature simplified — no longer requires nougat_model/processor args.
    """
    image_paths = pdf_to_images(pdf_path)
    all_text = []

    nougat_model, nougat_processor = _load_nougat()

    try:
        for img in image_paths:
            print(f"Scanning {img}...")
            text = run_nougat_ocr(img, nougat_model, nougat_processor)
            if not text:
                continue
            all_text.append(text)
            append_to_training_dataset(text)
    finally:
        _flush_nougat(nougat_model, nougat_processor)

    if all_text:
        combined = "\n\n".join(all_text)
        collection.add(
            documents=[f"Textbook content from {os.path.basename(pdf_path)}:\n{combined}"],
            metadatas=[{"memory_type": "raw_document", "source_file": os.path.basename(pdf_path)}],
            ids=[str(uuid.uuid4())]
        )
        print(f"[MEMORY] Full textbook saved to ChromaDB.")

def learn_from_images(image_paths, collection):
    """
    JIT loads Nougat, processes images, then immediately flushes VRAM.
    Signature simplified — no longer requires nougat_model/processor args.
    """
    all_extracted = []

    nougat_model, nougat_processor = _load_nougat()

    try:
        for img_path in image_paths:
            print(f"Scanning {img_path} with Nougat...")
            extracted_content = run_nougat_ocr(img_path, nougat_model, nougat_processor)
            if extracted_content:
                all_extracted.append(
                    f"--- Document: {os.path.basename(img_path)} ---\n{extracted_content}"
                )
                append_to_training_dataset(extracted_content)
    finally:
        _flush_nougat(nougat_model, nougat_processor)

    if not all_extracted:
        return ""

    combined_text = "\n\n".join(all_extracted)
    memory_text = f"User uploaded academic documents containing the following text and math:\n{combined_text}"

    collection.add(
        documents=[memory_text],
        metadatas=[{"memory_type": "raw_document"}],
        ids=[str(uuid.uuid4())]
    )

    return combined_text

def embed_and_save_text(raw_text, collection, source="Web Scrape"):
    """
    Takes pure string text (e.g., from Night Shift web scraping) and saves it directly to ChromaDB.
    Bypasses Nougat OCR entirely since the data is already text.
    """
    if not raw_text or not raw_text.strip():
        print(f"[MEMORY] Warning: No text provided to save from {source}.")
        return False

    memory_text = f"Research data from {source}:\n{raw_text}"

    try:
        collection.add(
            documents=[memory_text],
            metadatas=[{"memory_type": "autonomous_research", "source_file": source}],
            ids=[str(uuid.uuid4())]
        )
        print(f"[MEMORY] Successfully embedded and saved text from '{source}' to ChromaDB.")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save text to ChromaDB: {e}")
        return False
