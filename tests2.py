import pandas as pd
from main2 import FamilyClassifier
from main3 import (
    HuggingFacePipeline, PromptTemplate, LLMChain,
    tokenizer, model, pipeline,
    classification_rules,
    load_city_threshold_for_city,
    format_city_thresholds_for_prompt,
    format_classification_rules_for_prompt
)

# Создаём свой chain, как в main3
hf_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
llm = HuggingFacePipeline(pipeline=hf_pipeline)

system_prompt = """
You are an expert system that classifies families into income categories based on their parameters and city thresholds.

City data:
{city_thresholds}

Classification rules (conditions may contain expressions referencing city thresholds, e.g. "1.8 * Median" means 1.8 times the city's Median income):
{classification_rules}

Given a city and a family's parameters (Median, Mean, ALand, Stdev), classify the family into one of the categories above.
Explain which rules matched and how the decision was made.

Family data:
- City: {city}
- Median: {Median}
- Mean: {Mean}
- ALand: {ALand}
- Stdev: {Stdev}

Please provide the classification category and explanation.
"""

template = PromptTemplate(
    input_variables=["city", "Median", "Mean", "ALand", "Stdev", "city_thresholds", "classification_rules"],
    template=system_prompt
)

chain = LLMChain(llm=llm, prompt=template)


def nn_classify_family(family_data: dict) -> str:
    city = family_data.get("City", "")
    city_thresholds = load_city_threshold_for_city("city_thresholds.json", city)
    if not city_thresholds:
        return "Unknown", f"Город '{city}' не найден"

    inputs = { 
        "city": city,
        "Median": str(family_data.get("Median", "")),
        "Mean": str(family_data.get("Mean", "")),
        "ALand": str(family_data.get("ALand", "")),
        "Stdev": str(family_data.get("Stdev", "")),
        "city_thresholds": format_city_thresholds_for_prompt(city_thresholds),
        "classification_rules": format_classification_rules_for_prompt(classification_rules, city_thresholds)
    }

    response = chain.invoke(inputs)

    # Извлекаем текст
    if isinstance(response, dict):
        text = response.get("text", str(response))
    else:
        text = response

    # Пытаемся извлечь категорию
    first_line = text.strip().split('\n')[0] if isinstance(text, str) else ""
    category = first_line.split()[0] if first_line else ""

    return category, text


classifier = FamilyClassifier(
    city_thresholds_path='city_thresholds.json',
    rules_path='classification_rules.json'
)

test_cases = [
    ({'City': 'New York', 'Median': 200000, 'Mean': 220000, 'ALand': 3000, 'AWater': 100, 'Stdev': 0.4}, 'Обеспеченная'),
    ({'City': 'New York', 'Median': 100000, 'Mean': 105000, 'ALand': 1200, 'AWater': 50, 'Stdev': 0.9}, 'Средний класс'),
    ({'City': 'New York', 'Median': 60000, 'Mean': 70000, 'ALand': 700, 'AWater': 30, 'Stdev': 1.2}, 'Бедная'),
    ({'City': 'New York', 'Median': 25000, 'Mean': 30000, 'ALand': 200, 'AWater': 20, 'Stdev': 2.5}, 'За гранью бедности'),
    ({'City': 'New York', 'Median': 140000, 'Mean': 160000, 'ALand': 2500, 'AWater': 80, 'Stdev': 0.3}, 'Обеспеченная'),
    ({'City': 'New York', 'Median': 95000, 'Mean': 97000, 'ALand': 1000, 'AWater': 60, 'Stdev': 0.8}, 'Средний класс'),
    ({'City': 'New York', 'Median': 50000, 'Mean': 55000, 'ALand': 500, 'AWater': 40, 'Stdev': 1.5}, 'Бедная'),
    ({'City': 'New York', 'Median': 15000, 'Mean': 20000, 'ALand': 100, 'AWater': 10, 'Stdev': 3.0}, 'За гранью бедности'),
    ({'City': 'New York', 'Median': 180000, 'Mean': 195000, 'ALand': 3200, 'AWater': 150, 'Stdev': 0.2}, 'Обеспеченная'),
    ({'City': 'New York', 'Median': 80000, 'Mean': 90000, 'ALand': 1100, 'AWater': 70, 'Stdev': 0.7}, 'Средний класс')
]

print("Результаты тестирования:\n")
for i, (family_data, expected) in enumerate(test_cases, 1):
    family_series = pd.Series(family_data)

    classic_result = classifier.classify_family(family_series)

    try:
        nn_category, nn_full_text = nn_classify_family(family_data)
    except Exception as e:
        nn_category = f"NN classification failed: {e}"
        nn_full_text = ""

    print(f"Тест {i}:")
    print(f"  Ожидалось:        {expected}")
    print(f"  Классический:     {classic_result}")
    print(f"  Нейросеть категория: {nn_category}")
    print(f"  Нейросеть полный ответ:\n{nn_full_text}\n")
    print(f"  Совпадение:       {classic_result == nn_category}\n")
