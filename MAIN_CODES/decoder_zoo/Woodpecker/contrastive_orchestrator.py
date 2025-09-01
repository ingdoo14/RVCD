import os
from typing import Tuple, Dict, Any
from PIL import Image
import numpy as np

from MAIN_CODES.YOLO.yolo import load_yolo_model, run_inference


def save_rgb_image(array: np.ndarray, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    Image.fromarray(array).save(path)


def vlm_infer_blip2(processor, model, image_rgb: np.ndarray, instruction: str) -> str:
    img = Image.fromarray(image_rgb)
    inputs = processor(img, instruction, return_tensors="pt").to("cuda", model.config.torch_dtype)
    generated_ids = model.generate(**inputs)
    text = processor.decode(generated_ids[0], skip_special_tokens=True)
    return text.strip()


def contrastive_merge(text_a: str, text_b: str) -> str:
    if text_a == text_b:
        return text_a
    if len(text_b) > len(text_a) and text_b.lower() not in text_a.lower():
        return text_b
    return text_a


def run_contrastive_pipeline(
    image_path: str,
    instruction: str,
    yolo_model_path: str = "yolov8n.pt",
    postprocess: str = "blur",  # or "mask"/"mosaic"
    conf: float = 0.25,
    blur_ksize: int = 31,
    mosaic_block_size: int = 15,
    vlm_processor=None,
    vlm_model=None,
    save_debug_dir: str = None,
) -> Dict[str, Any]:
    yolo = load_yolo_model(yolo_model_path)
    bboxes, probs, names, img_rgb_processed = run_inference(
        yolo, image_path, postprocess=postprocess, blur_ksize=blur_ksize, mosaic_block_size=mosaic_block_size, conf=conf
    )
    # also get original image without postprocess
    _, _, _, img_rgb_orig = run_inference(yolo, image_path, postprocess=None, conf=conf)

    if save_debug_dir:
        save_rgb_image(img_rgb_orig, os.path.join(save_debug_dir, "orig.jpg"))
        save_rgb_image(img_rgb_processed, os.path.join(save_debug_dir, f"{postprocess}.jpg"))

    # VLM inference on both images
    assert vlm_processor is not None and vlm_model is not None, "Provide VLM processor and model"
    text_orig = vlm_infer_blip2(vlm_processor, vlm_model, img_rgb_orig, instruction)
    text_proc = vlm_infer_blip2(vlm_processor, vlm_model, img_rgb_processed, instruction)

    merged = contrastive_merge(text_orig, text_proc)

    return {
        "bboxes": bboxes,
        "probs": probs,
        "names": names,
        "text_original": text_orig,
        "text_processed": text_proc,
        "merged_text": merged,
    }


if __name__ == "__main__":
    import argparse
    import torch
    from transformers import Blip2Processor, Blip2ForConditionalGeneration

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--yolo", default="yolov8n.pt")
    parser.add_argument("--postprocess", default="blur")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--save_debug_dir", default=None)
    args = parser.parse_args()

    processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xxl")
    device_map = {"query_tokens": 0, "vision_model": 0, "language_model": 0, "language_projection": 0, "qformer": 0}
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-flan-t5-xxl", load_in_8bit=True, device_map=device_map, torch_dtype=torch.float16
    )

    out = run_contrastive_pipeline(
        image_path=args.image,
        instruction=args.instruction,
        yolo_model_path=args.yolo,
        postprocess=args.postprocess,
        conf=args.conf,
        vlm_processor=processor,
        vlm_model=model,
        save_debug_dir=args.save_debug_dir,
    )
    print(out["merged_text"]) 
