
import os
import json
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# Setup DeepSeek
llm = ChatOpenAI(
    model="deepseek-chat", # or 'deepseek-v3'
    openai_api_key=os.environ.get("DEEPSEEK_API_KEY"),
    openai_api_base="https://api.deepseek.com",
    temperature=0.7
)

def generate_questions():
    print("🤖 Generating 10 comparison questions using DeepSeek...")
    
    prompt = """
    You are an expert automotive analyst.
    I need you to generate 10 difficult and specific questions to test a RAG (Retrieval Augmented Generation) system.
    
    The RAG system has two documents:
    1. "StarEra ES9" (Electric SUV)
    2. "Xiaomi SU7" (Electric Sedan)
    
    Requirements for Questions:
    - 3 Questions about ES9 specifics (e.g., specific range, battery size, dimensions).
    - 3 Questions about SU7 specifics (e.g., acceleration, motor power, charging).
    - 4 Questions comparing both (e.g., "Which car has longer range, ES9 or SU7?", "Compare the intelligent driving chips of ES9 and SU7").
    
    Format essential:
    Return a JSON array of objects. Each object must have:
    - "question": The question string.
    - "reference_answer": A concise correct answer containing the key facts/numbers.
    - "key_facts": A list of keywords/numbers that MUST appear in the retrieved documents for it to be considered a 'hit'.
    
    Example Schema:
    [
        {
            "question": "What is the max CLTC range of the Xiaomi SU7?",
            "reference_answer": "The Xiaomi SU7 Max has a CLTC range of 800km (or 810km depending on tires).",
            "key_facts": ["800km", "CLTC", "SU7"]
        },
        ...
    ]
    
    Please output ONLY the JSON string.
    """
    
    try:
        response = llm.invoke(prompt)
        content = response.content
        
        # Cleanup JSON formatting
        content = content.replace("```json", "").replace("```", "").strip()
        
        questions = json.loads(content)
        
        # Save to file
        output_file = "benchmark/questions.json"
        with open(output_file, "w") as f:
            json.dump(questions, f, indent=4, ensure_ascii=False)
            
        print(f"✅ Generated {len(questions)} questions. Saved to {output_file}")
        
    except Exception as e:
        print(f"Error generating questions: {e}")

if __name__ == "__main__":
    generate_questions()
