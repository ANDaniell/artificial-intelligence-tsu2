#!/usr/bin/env python3
# vacation_expert_final.py

import os
import json
import re
import torch
from typing import List, Dict, Any
from nltk.stem import PorterStemmer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline as hf_pipeline
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
MODEL_ID       = "tiiuae/falcon-7b-instruct"
EMBED_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"
KB_PATH        = "../../Downloads/Telegram Desktop/holidays.json"
DB_DIR         = "./vacation_rag_db"
TOP_K          = 3
MAX_NEW_TOKENS = 200
TEMPERATURE    = 0.7
TOP_P          = 0.9
# ────────────────────────────────────────────────────────────────────────────────

# ── STEMMER ─────────────────────────────────────────────────────────────────────
stemmer = PorterStemmer()

def match_keyword(keyword: str, text: str) -> bool:
    if not keyword:
        return True
    root = stemmer.stem(keyword.lower())
    tokens = re.findall(r"\b\w+\b", text.lower())
    return any(root == stemmer.stem(tok) for tok in tokens)

# ── LOAD OPTIONS ───────────────────────────────────────────────────────────────
def load_options(path: str = KB_PATH) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)

# ── MATCHING LOGIC ─────────────────────────────────────────────────────────────
def option_matches(opt: Dict[str, Any], crit: Dict[str, Any]) -> bool:
    if crit.get("country") and opt["country"].lower() != crit["country"].lower():
        return False
    if crit.get("exclusive") and not opt.get("exclusive", False):
        return False
    if opt.get("stars", 0) < crit.get("min_stars", 0):
        return False
    if crit.get("breakfast") and not opt.get("breakfast_included", False):
        return False
    if crit.get("lunch") and not opt.get("lunch_included", False):
        return False
    if crit.get("dinner") and not opt.get("dinner_included", False):
        return False
    if not match_keyword(crit.get("keyword", ""), opt.get("comment", "")):
        return False
    max_price = crit.get("max_price")
    if max_price is not None and opt.get("price", float("inf")) > max_price:
        return False
    for am in crit.get("amenities", []):
        if am.lower() not in [a.lower() for a in opt.get("amenities", [])]:
            return False
    max_dist = crit.get("max_distance")
    if max_dist is not None and opt.get("distance", float("inf")) > max_dist:
        return False
    return True

# ── INITIALIZE LLM & VECTORSTORE ───────────────────────────────────────────────
print(f"Loading LLM {MODEL_ID}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    low_cpu_mem_usage=False,
)
pipe = hf_pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
    max_new_tokens=MAX_NEW_TOKENS,
    do_sample=True,
    temperature=TEMPERATURE,
    top_p=TOP_P,
)
llm = HuggingFacePipeline(pipeline=pipe)
print("LLM ready.\n")

options = load_options()
docs = []
for o in options:
    docs.append(
        Document(
            page_content=(
                f"Name: {o['name']}\n"
                f"Country: {o['country']} | Stars: {o['stars']} | Price: {o['price']}\n"
                f"Amenities: {', '.join(o['amenities'])}\n"
                f"Comment: {o['comment']}\n"
            ),
            metadata={"name": o['name']},
        )
    )

vectordb = Chroma.from_documents(
    docs,
    HuggingFaceEmbeddings(model_name=EMBED_MODEL),
    persist_directory=DB_DIR
)

rag_prompt = PromptTemplate.from_template(
    """You are a travel advisor. Based on the hotel descriptions below, answer the question.

{context}

Q: {question}
A:"""
)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectordb.as_retriever(search_kwargs={"k": TOP_K}),
    chain_type_kwargs={"prompt": rag_prompt},
    return_source_documents=True
)

# ── MODES ──────────────────────────────────────────────────────────────────────
def run_holiday_mode():
    crit = {
        'country': input('Country (blank=any): ').strip(),
        'exclusive': input('Exclusive only (yes/no): ').lower().startswith('y'),
        'min_stars': float(input('Min stars [3.0]: ').strip() or 3.0),
        'breakfast': input('Require breakfast (yes/no): ').lower().startswith('y'),
        'lunch': input('Require lunch (yes/no): ').lower().startswith('y'),
        'dinner': input('Require dinner (yes/no): ').lower().startswith('y'),
        'keyword': input('Keyword (blank=none): ').strip(),
        'max_price': (lambda x: float(x) if x else None)(input('Max price (blank=none): ').strip()),
        'amenities': [a.strip() for a in input('Amenities (comma-separated): ').split(',') if a.strip()],
        'max_distance': (lambda x: float(x) if x else None)(input('Max distance km (blank=none): ').strip())
    }
    matched = [o for o in options if option_matches(o, crit)]
    if not matched:
        print('No hotels match your criteria.')
        return
    print(f"Found {len(matched)} hotels. Generating recommendation...\n")
    expert_prompt = PromptTemplate(
        input_variables=['criteria','options'],
        template=(
            'You are an expert travel advisor.\n'
            'Criteria: {criteria}\n'
            'Options: {options}\n'
            'Provide a ranked recommendation and vacation tips.'
        )
    )
    prompt_in = expert_prompt.format(
        criteria=json.dumps(crit, indent=2),
        options=json.dumps(matched, indent=2)
    )
    rec = llm.invoke(prompt_in)
    print(rec)


def run_rag_mode():
    query = input('Ask about your vacation options: ').strip()
    res = qa_chain.invoke(query)
    print("\nAnswer:\n", res['result'])
    if res.get('source_documents'):
        print("\nSources:")
        seen = set()
        for doc in res['source_documents']:
            name = doc.metadata.get('name')
            if name not in seen:
                seen.add(name)
                print(f"- {name}")


def main():
    print('''\n=== Free Vacation Expert + RAG System ===\n1) Holiday Expert Filter & Chat\n2) RAG-powered Vacation Q&A\n''')
    choice = input('Select [1/2]: ').strip()
    if choice == '1':
        run_holiday_mode()
    elif choice == '2':
        run_rag_mode()
    else:
        print('Invalid choice.')

if __name__ == '__main__':
    main()
