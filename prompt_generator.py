import json
import os
from datetime import datetime
from langchain_core.prompts import PromptTemplate

PROMPT_FILE = "saved_prompts.json"


def get_prompt_template():
    return PromptTemplate(
        input_variables=["paper", "style", "length"],
        template="""
Explain the research paper titled "{paper}".

Explanation style: {style}
Explanation length: {length}

Make the explanation clear, structured, and accurate.
"""
    )


def save_prompt_to_json(paper, style, length, final_prompt):
    prompt_entry = {
        "paper": paper,
        "style": style,
        "length": length,
        "prompt": final_prompt,
        "timestamp": datetime.now().isoformat()
    }

    # Load existing prompts
    if os.path.exists(PROMPT_FILE):
        with open(PROMPT_FILE, "r") as f:
            data = json.load(f)
    else:
        data = []

    data.append(prompt_entry)

    # Save back
    with open(PROMPT_FILE, "w") as f:
        json.dump(data, f, indent=4)

    return prompt_entry
