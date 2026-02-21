import os
import sys
import json
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field

# Load environment variables (GEMINI_API_KEY)
load_dotenv()

# 1. Define the exact JSON structure we need to output
class GreenPullRequest(BaseModel):
    pr_title: str = Field(description="A catchy title for the Pull Request, e.g., '🌱 GreenPull: Implement QLoRA'")
    pr_body: str = Field(description="Markdown body explaining the sustainability benefits and estimated savings")
    refactored_code: str = Field(description="The complete, fully runnable refactored Python code. NO markdown formatting.")
    quantization_applied: str = Field(description="The level of quantization applied, e.g., 'None', '8-bit', '4-bit'")
    technique_used: str = Field(description="The optimization technique used, e.g., 'LoRA', 'FP16', 'Batch Size Tuning'")

def analyze_and_refactor(code_string: str) -> GreenPullRequest:
    """Takes raw Python code, sends it to Gemini, and returns structured PR data."""
    client = genai.Client()
    
    prompt = f"""
    You are 'GreenPull', an expert AI Sustainability Engineer. 
    Your goal is to optimize the provided machine learning code to minimize its carbon footprint, energy consumption, and hardware requirements.
    
    Instructions:
    1. Analyze the code for inefficiencies (e.g., full-parameter fine-tuning, FP32 inference, large unoptimized models).
    2. Refactor the code to implement Green AI best practices (e.g., inject bitsandbytes INT8/INT4 quantization, implement PEFT/LoRA adapters).
    3. Ensure the core business logic and model architecture remain intact—only change the training/inference mechanisms.
    4. Write a professional GitHub Pull Request title and body explaining the exact changes and why they reduce CO2e emissions.
    
    CRITICAL CODING STANDARDS:
    - DO NOT use `bitsandbytes` (no `load_in_4bit` or `load_in_8bit`). It causes deep architectural crashes on older encoder models during the backward pass.
    - INSTEAD, achieve Green AI energy savings by implementing Parameter-Efficient Fine-Tuning (PEFT) with LoRA.
    - Combine LoRA with PyTorch Automatic Mixed Precision (AMP) using `torch.amp.autocast('cuda')` and `torch.amp.GradScaler('cuda')` in the training loop.
    - Ensure the model is properly moved to the device using `model.to(device)` before training begins..

    Here is the original code to refactor:
    {code_string}
    """

    response = client.models.generate_content(
        model='gemini-2.5-flash', 
        contents=prompt,
        config={
            'response_mime_type': 'application/json',
            'response_schema': GreenPullRequest,
            'temperature': 0.1, 
        },
    )
    return response.parsed

if __name__ == "__main__":
    # --- INTERFACE WITH PERSON 1 ---
    # We expect Person 1 to save the raw GitHub code into a file named "extracted_code.py"
    input_filename = "test_extracted_code.py"
    output_filename = "greenpull_pr_data.json"
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_filepath = os.path.join(script_dir, input_filename)
    output_filepath = os.path.join(script_dir, output_filename)

    # 1. Check if Person 1's output exists
    if not os.path.exists(input_filepath):
        print(f"❌ Error: Cannot find '{input_filename}'.")
        print("Please ensure Person 1's script has run and saved the extracted code to this directory.")
        sys.exit(1)

    # 2. Read the code Person 1 extracted
    print(f"📥 Reading extracted code from {input_filename}...")
    with open(input_filepath, "r", encoding="utf-8") as f:
        extracted_code = f.read()

    # 3. Pass it to the AI Agent
    print("🧠 Sending code to Gemini for Green AI optimization...")
    try:
        result = analyze_and_refactor(extracted_code)
        
        # 4. Save the JSON output for Person 1 (to open the PR) and Person 3 (for analytics)
        with open(output_filepath, "w", encoding="utf-8") as f:
            json.dump(result.model_dump(), f, indent=2)
            
        print(f"\n✅ Optimization complete!")
        print(f"💾 Success! Structured JSON saved to: {output_filepath}")
        print(f"📌 PR Title Preview: {result.pr_title}")
        print(f"🛠️ Technique Used: {result.technique_used}")
        print("\nReady for Person 1 to open the PR and Person 3 to calculate CodeCarbon stats!")
        
    except Exception as e:
        print(f"\n❌ Error during AI generation: {e}")