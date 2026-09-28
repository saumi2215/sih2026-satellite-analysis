from PIL import Image
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    BlipForQuestionAnswering,
)
 
# ---------- 1. Load models (done once, reused for every call) ----------
 
print("Loading BLIP models... (first time only, this may take a minute)")
 
_caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
_caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
 
_vqa_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
_vqa_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
 
print("Models loaded.")
 
 
# ---------- 2. Captioning (describe the image) ----------
 
def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = _caption_processor(image, return_tensors="pt")
    out = _caption_model.generate(**inputs, max_new_tokens=30)
    caption = _caption_processor.decode(out[0], skip_special_tokens=True)
    return caption
 
 
# ---------- 3. VQA (answer a specific question) ----------
 
def answer_question(image_path, question):
    image = Image.open(image_path).convert("RGB")
    inputs = _vqa_processor(image, question, return_tensors="pt")
    out = _vqa_model.generate(**inputs, max_new_tokens=20)
    answer = _vqa_processor.decode(out[0], skip_special_tokens=True)
    return answer
 
 
# ---------- 4. Main entry point -> Output to Module 7 ----------
 
def run_vqa_captioning(image_path, question=None):
    """
    Main function M3 (AI Agent) calls when it routes a task to Module 4.
 
    If a question is given -> VQA mode (answers that specific question).
    If no question is given -> Captioning mode (describes the image).
 
    Returns a dict forwarded to Module 7:
        {
            'mode': 'vqa' or 'caption',
            'question': str or None,
            'answer': str,
            'confidence': float
        }
    """
    result = {}
 
    if question and question.strip():
        answer = answer_question(image_path, question)
        result['mode'] = 'vqa'
        result['question'] = question
        result['answer'] = answer
    else:
        caption = generate_caption(image_path)
        result['mode'] = 'caption'
        result['question'] = None
        result['answer'] = caption
 
    result['confidence'] = 0.8  # placeholder; BLIP doesn't give a direct confidence score
    return result
 
 
if __name__ == "__main__":
    # Example usage - replace with your actual processed image path
    image_path = "m2_outputs/processed.png"
 
    # Test 1: Captioning
    print("\n--- Captioning ---")
    print(run_vqa_captioning(image_path))
 
    # Test 2: VQA
    print("\n--- VQA ---")
    print(run_vqa_captioning(image_path, question="What is shown in this image?"))
 