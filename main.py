from module6 import analyze_optical_sar
from module5 import run_grounding_change_detection
from module4 import run_vqa_captioning
 
 
# ---------- Module 3 - AI Agent (same logic as your module3 file) ----------
 
def ai_agent(question):
    question = question.lower()
 
    if "change" in question or "difference" in question:
        module = "M5"
        task = "Detect changes between images"
 
    elif "sar" in question or "optical" in question:
        module = "M6"
        task = "Perform Optical + SAR analysis"
 
    else:
        module = "M4"
        task = "Answer question or describe the image"
 
    return module, task
 
 
# ---------- Integration logic ----------
 
def route_and_execute(question, optical_path=None, sar_path=None,
                       optical_t2_path=None, sar_t2_path=None):
    module, task = ai_agent(question)
 
    print("\n--- M3 Result ---")
    print("Selected Module:", module)
    print("Task:", task)
 
    if module == "M6":
        if not optical_path or not sar_path:
            print("\n[Error] M6 needs both optical_path and sar_path to run.")
            return None
 
        print("\n--- Running M6 (Optical + SAR Analysis) ---")
        result = analyze_optical_sar(
            optical_path=optical_path,
            sar_path=sar_path,
            optical_t2_path=optical_t2_path,
            sar_t2_path=sar_t2_path
        )
        print("\n--- M6 Result ---")
        print(result)
        return result
 
    elif module == "M5":
        if not optical_path:
            print("\n[Error] M5 needs at least one image (optical_path) to run.")
            return None
 
        print("\n--- Running M5 (Grounding & Change Detection) ---")
        result = run_grounding_change_detection(
            task=task,
            image_path=optical_path,
            image_t2_path=optical_t2_path  # if given, runs change-detection mode; else grounding mode
        )
        print("\n--- M5 Result ---")
        print(result)
        return result
 
    else:  # M4
        if not optical_path:
            print("\n[Error] M4 needs an image (optical_path) to run.")
            return None
 
        print("\n--- Running M4 (VQA & Captioning) ---")
        # If the question is just "describe/what is this image" style, treat as captioning (no question passed).
        # Otherwise pass the actual question through for VQA.
        is_generic_describe = any(word in question.lower() for word in ["describe", "caption"])
        result = run_vqa_captioning(
            image_path=optical_path,
            question=None if is_generic_describe else question
        )
        print("\n--- M4 Result ---")
        print(result)
        return result
 
 
if __name__ == "__main__":
    user_question = input("Enter your question: ")
 
    # For now, hardcode test image paths (same ones you already generated)
    result = route_and_execute(
        question=user_question,
        optical_path="optical_t1.jpg",
        sar_path="sar_t1.jpg",
        optical_t2_path="optical_t2.jpg"  # needed for M5 change-detection / M6 multi-temporal mode
    )
 