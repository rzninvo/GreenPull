import json
import os

def extract_optimized_code():
    # 1. Define your file names
    input_json_name = "greenpull_pr_data.json" # Change to mock_handoff.json if that's what you used
    output_python_name = "optimized_training.py"

    # 2. Get absolute paths to avoid directory confusion
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, input_json_name)
    output_path = os.path.join(script_dir, output_python_name)

    print(f"🔍 Looking for AI output in: {input_json_name}...")

    try:
        # 3. Open and parse the JSON file
        with open(json_path, "r", encoding="utf-8") as f:
            pr_data = json.load(f)

        # 4. Extract the exact code string
        refactored_code = pr_data.get("refactored_code")
        
        if not refactored_code:
            print("❌ Error: Could not find 'refactored_code' inside the JSON file.")
            return

        # 5. Write the extracted string into a proper .py file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(refactored_code)

        print("✅ Extraction Complete!")
        print(f"💾 Your fully formatted, runnable Python file is saved at: {output_path}")
        print("Ready for Person 3 to run CodeCarbon analytics!")

    except FileNotFoundError:
        print(f"❌ Error: Cannot find {json_path}. Did you run the AI Agent first?")
    except json.JSONDecodeError:
        print(f"❌ Error: {json_path} contains invalid JSON.")

if __name__ == "__main__":
    extract_optimized_code()
