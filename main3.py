import json
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, AutoModelForCausalLM
import torch
import re
from transformers import BitsAndBytesConfig


model_name = "mistralai/Mistral-7B-Instruct-v0.2"  # ⚠️ требует ручного доступа, см. ниже
device = "cuda" if torch.cuda.is_available() else "cpu"

# Квантование в 4 бита через BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# Загружаем токенизатор
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Загружаем модель (через quantization_config)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config if device == "cuda" else None,
    device_map="auto"
)

# Создаем pipeline (без аргумента device!)
hf_pipeline = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    temperature=0.7,
    max_new_tokens=528,
    return_full_text=False,
    do_sample=True
)


# Обертка для LangChain
from langchain_community.llms import HuggingFacePipeline
llm = HuggingFacePipeline(pipeline=hf_pipeline)


'''
model_name = "google/flan-t5-large"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None
)

hf_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
llm = HuggingFacePipeline(pipeline=hf_pipeline)
'''

def load_city_threshold_for_city(path: str, city: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        all_data = json.load(f)
    return all_data.get(city, {})


def load_classification_rules(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_expression(expr: str, city_params: dict) -> float:
    def repl(match):
        key = match.group(0)
        return str(city_params.get(key, 0))

    expr_clean = re.sub(r'\b[A-Za-z]+\b', repl, expr)
    try:
        return eval(expr_clean)
    except Exception:
        return 0.0


def format_city_thresholds_for_prompt(city_thresholds: dict) -> str:
    lines = ["City thresholds:"]
    for param, value in city_thresholds.items():
        lines.append(f"- {param}: {value}")
    return "\n".join(lines)


def format_classification_rules_for_prompt(rules: dict, city_params: dict) -> str:
    lines = ["Classification rules:"]
    for category, rule_sets in rules.items():
        lines.append(f"Category '{category}' if any of:")
        for rule_set in rule_sets:
            conds = []
            for cond in rule_set:
                field, op, val_expr = cond
                val = val_expr
                if isinstance(val_expr, str) and any(c.isalpha() for c in val_expr):
                    val = evaluate_expression(val_expr, city_params)
                    val = round(val, 3)
                conds.append(f"{field} {op} {val}")
            lines.append(" AND ".join(conds))
        lines.append("")
    return "\n".join(lines)


classification_rules = load_classification_rules("classification_rules.json")


def classify_family_with_llm(city: str, median: str, mean: str, aland: str, stdev: str) -> str:
    city_thresholds = load_city_threshold_for_city("city_thresholds.json", city)
    if not city_thresholds:
        print(f"Город '{city}' не найден в базе данных.")
        return f"Город '{city}' не найден в базе данных."

    city_thresholds_text = format_city_thresholds_for_prompt(city_thresholds)
    classification_rules_text = format_classification_rules_for_prompt(classification_rules, city_thresholds)
    # print(city_thresholds_text)
    system_prompt = f"""
    You are an expert system that classifies families into income categories based on their parameters and city thresholds.
    City data: {city_thresholds_text}
    Classification rules (conditions may contain expressions referencing city thresholds, e.g. "1.8 * Median" means 1.8 times the city's Median income):
    {classification_rules_text}
    Given a city and a family's parameters (Median, Mean, ALand, Stdev), classify the family into one of these categories:
    Wealthy, Middle class, Poor, Below poverty line.
    Respond in exactly the following format (without quotes):
    Do not add any other text.
    You MUST start your reply with the line:
    Category: <category name>

    Then write:
    Explanation: <your reasoning>

    Do not start your reply with any other text. Do not explain what you're doing. Do not add headers like 'Family data:' or 'Classification rules:'.
    """

    template = PromptTemplate(
        input_variables=["city", "Median", "Mean", "ALand", "Stdev"],
        template=system_prompt + """

Family data:
- City: {city}
- Median: {Median}
- Mean: {Mean}
- ALand: {ALand}
- Stdev: {Stdev}
"""
    )

    chain = template | llm

    response = chain.invoke({
        "city": city,
        "Median": median,
        "Mean": mean,
        "ALand": aland,
        "Stdev": stdev
    })

    return response


if __name__ == "__main__":
    city = input("Введите город: ").strip()
    median = input("Введите Median семьи: ").strip()
    mean = input("Введите Mean семьи: ").strip()
    aland = input("Введите ALand семьи: ").strip()
    stdev = input("Введите Stdev семьи: ").strip()

    result = classify_family_with_llm(city, median, mean, aland, stdev)
    print("\nРезультат классификации:\n", result)
