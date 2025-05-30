import json
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from langchain_community.llms import HuggingFacePipeline
import torch
import re

# Конфигурация модели
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Настройка квантования в 4 бита (если доступно)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# Загрузка токенизатора и модели
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config if device == "cuda" else None,
    device_map="auto"
)

# Создание pipeline и LangChain-обёртки
hf_pipeline = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    temperature=0.3,
    max_new_tokens=528,
    return_full_text=False,
    do_sample=True
)
llm = HuggingFacePipeline(pipeline=hf_pipeline)


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
    return "\n".join(["City thresholds:"] + [f"- {param}: {value}" for param, value in city_thresholds.items()])


def format_classification_rules_for_prompt(rules: dict, city_params: dict) -> str:
    lines = ["Classification rules:"]
    for category, rule_sets in rules.items():
        lines.append(f"Category '{category}' if any of:")
        for rule_set in rule_sets:
            conds = []
            for cond in rule_set:
                field, op, val_expr = cond
                val = evaluate_expression(val_expr, city_params) if isinstance(val_expr, str) and any(c.isalpha() for c in val_expr) else val_expr
                conds.append(f"{field} {op} {round(val, 3) if isinstance(val, float) else val}")
            lines.append(" AND ".join(conds))
        lines.append("")
    return "\n".join(lines)


classification_rules = load_classification_rules("classification_rules.json")


def classify_family_with_llm(city: str, median: str, mean: str, aland: str, stdev: str) -> str:
    city_thresholds = load_city_threshold_for_city("city_thresholds.json", city)
    if not city_thresholds:
        return f"Город '{city}' не найден в базе данных."

    city_thresholds_text = format_city_thresholds_for_prompt(city_thresholds)
    classification_rules_text = format_classification_rules_for_prompt(classification_rules, city_thresholds)

    system_prompt = f"""
    You are an expert system that classifies families into income categories.
    
    - "City Median" means the city's median income.
    - "Family Median" means the family's income.

    City thresholds:
    {city_thresholds_text}
    
    When comparing thresholds, always interpret expressions like "1.8 * Median" as referring to **city thresholds**. 
Family data will always use the prefix "Family", e.g., "Family Median".

    Classification rules (expressed using city thresholds like '0.6 * City Median'):

    {classification_rules_text}

    Given family data (Family Median, Family Mean, etc.), determine the appropriate category.
    ...
    """

    template = PromptTemplate(
        input_variables=["city", "Median", "Mean", "ALand", "Stdev"],
        template=system_prompt + """
Family data:
- City: {city}
- Family Median: {Median}
- Family Mean: {Mean}
- Family ALand: {ALand}
- Family Stdev: {Stdev}
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
