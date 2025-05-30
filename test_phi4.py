from langchain_huggingface import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
import json

model_name = "google/flan-t5-large"

# Проверяем доступность CUDA
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device set to use {device}")
if device == "cuda":
    print("GPU name:", torch.cuda.get_device_name(0))
    print("Memory allocated:", round(torch.cuda.memory_allocated(0) / 1024 ** 3, 2), "GB")

# Загружаем токенизатор и модель с оптимизациями
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Создаем HuggingFace pipeline
hf_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# Создаем LangChain LLM через langchain-huggingface
llm = HuggingFacePipeline(pipeline=hf_pipeline)

def load_knowledge(json_path: str) -> str:
    with open(json_path, "r", encoding="utf-8") as f:
        rules = json.load(f)

    prompt_lines = ["You are an assistant that classifies families into income categories based on certain rules.\n"]
    prompt_lines.append("The rules are as follows:\n")

    for category, rule_sets in rules.items():
        prompt_lines.append(f"If family meets one of the following conditions, it is classified as '{category}':")
        for rule_set in rule_sets:
            if isinstance(rule_set[0], list):
                conditions = " AND ".join(
                    f"{field} {op} {value}" for (field, op, value) in rule_set
                )
            else:
                field, op, value = rule_set
                conditions = f"{field} {op} {value}"
            prompt_lines.append(f"- {conditions}")
        prompt_lines.append("")  # blank line

    return "\n".join(prompt_lines)

# Загружаем системный промпт с правилами из JSON
system_prompt = load_knowledge("rules.json")

# Создаем шаблон с переменными
template = PromptTemplate(
    input_variables=["budget", "purpose", "priority"],
    template=system_prompt + """

Based on the user's preferences:
- Budget: {budget}
- Usage Purpose: {purpose}
- Priority: {priority}

Please recommend a suitable car model. Justify the recommendation using the classification rules if relevant.
"""
)

# Создаем цепочку LangChain
chain = template | llm

if __name__ == "__main__":
    budget = input("Enter your budget: ")
    purpose = input("What is the main purpose of the vehicle (e.g., city commuting, off-road, family travel)? ")
    priority = input("What is your top priority (e.g., fuel efficiency, spaciousness, speed)? ")

    response = chain.invoke({"budget": budget, "purpose": purpose, "priority": priority})
    print("\nModel Recommendation:\n", response)
