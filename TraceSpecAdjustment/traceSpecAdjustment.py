from openai import OpenAI
from dotenv import load_dotenv
import json

load_dotenv()

openai_client = OpenAI()


def analyze_requirement_changes(old_description: str, new_description: str) -> dict:
    """
    Analyze changes between two Jira ticket descriptions and return a structured JSON.

    Args:
        old_description (str): Previous ticket description.
        new_description (str): Updated ticket description.

    Returns:
        dict: JSON containing a summary of changes, detailed change information,
              and the original descriptions.
    """

    prompt = f"""
    You are a project management and software requirements analysis expert.

    Analyze the changes between the old and new Jira ticket descriptions below,
    and simulate what the client now wants based on the new description.

    Old description:
    {old_description}

    New description:
    {new_description}

    Tasks:
    1. Identify precisely what has changed between the old and new description.
    2. Summarize the changes concisely **from the client's perspective**.
       Use phrases like "The client now wants..." to reflect intent.
    3. Categorize changes by type if applicable: "Added feature", "Modified feature", "Removed feature", "Priority change", "Technical detail change".
    4. Provide recommendations or key points for the development team if relevant.
    5. Respond in french if the descriptions are in french.

    Return the result in JSON format exactly like this:

    {{
        "summary_changes": "Concise summary of what the client now wants",
        "changes_details": [
            {{
                "type": "Type of change (Added/Modified/Removed/Detail/Priority)",
                "description": "Detailed description of the change"
            }}
        ],
        "recommendations": "Suggestions or important points for the team"
    }}
    """

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert in software project requirements and Jira tickets."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    answer = response.choices[0].message.content

    try:
        result_json = json.loads(answer)
    except:
        result_json = {"raw_text": answer}

    # Ajouter les descriptions originales dans le JSON
    result_json["old_description"] = old_description
    result_json["new_description"] = new_description

    return result_json


# --- Example usage ---
if __name__ == "__main__":
    old_desc = "L'utilisateur peut créer un compte et se connecter par e-mail."
    new_desc = "L'utilisateur peut créer un compte, se connecter par e-mail ou via Google, et réinitialiser son mot de passe."

    changes = analyze_requirement_changes(old_desc, new_desc)
    print(json.dumps(changes, indent=2, ensure_ascii=False))
