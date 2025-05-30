import json
import re
import pandas as pd
from typing import Optional, Dict

class FamilyClassifier:
    def __init__(self, city_thresholds_path: str, rules_path: str):
        with open(city_thresholds_path, 'r', encoding='utf-8') as f:
            self.city_stats = json.load(f)

        with open(rules_path, 'r', encoding='utf-8') as f:
            self.rules = json.load(f)

    def _get_city_thresholds(self, city: str) -> Optional[Dict]:
        return self.city_stats.get(city)

    def _evaluate_expression(self, expr: str, thresholds: Dict) -> float:
        """
        Вычисляет выражение вида "2.0 * Median", "Median", "100000" и т.п.
        thresholds - словарь с порогами города, ключи: Median, Mean, ALand, Stdev
        """
        expr = expr.strip()
        # Паттерн для выражения с множителем: число * параметр
        pattern = r"([\d\.]+)\s*\*\s*([A-Za-z]+)"
        match = re.fullmatch(pattern, expr)
        if match:
            multiplier = float(match.group(1))
            field = match.group(2)
            if field in thresholds:
                return multiplier * thresholds[field]
            else:
                return float('inf')
        elif expr in thresholds:
            return thresholds[expr]
        else:
            try:
                return float(expr)
            except ValueError:
                return float('inf')

    def _evaluate_condition(self, family: pd.Series, thresholds: Dict, condition: list) -> bool:
        """
        condition: [field:str, operator:str, expr:str]
        family: pd.Series с параметрами семьи (Median, Mean, ALand, Stdev)
        thresholds: dict с порогами города
        """
        field, op, expr = condition
        family_val = family.get(field)
        if family_val is None:
            return False

        threshold_val = self._evaluate_expression(expr, thresholds)
        # Для отладки:
        # print(f"DEBUG: {field} {op} {expr} -> family_val={family_val}, threshold_val={threshold_val}")

        if op == '>':
            return family_val > threshold_val
        elif op == '<':
            return family_val < threshold_val
        elif op == '>=':
            return family_val >= threshold_val
        elif op == '<=':
            return family_val <= threshold_val
        elif op == '==':
            return family_val == threshold_val
        elif op == '!=':
            return family_val != threshold_val
        else:
            return False

    def _check_rule_set(self, family: pd.Series, thresholds: Dict, rule_set: list) -> bool:
        return all(self._evaluate_condition(family, thresholds, cond) for cond in rule_set)

    def classify_family(self, family: pd.Series) -> str:
        thresholds = self._get_city_thresholds(family['City'])
        if not thresholds:
            return "Недостаточно данных по городу"

        for category, rule_sets in self.rules.items():
            for rule_set in rule_sets:
                if self._check_rule_set(family, thresholds, rule_set):
                    print(f"DEBUG: Семья классифицирована как '{category}' по правилу {rule_set}")
                    return category

        return "Не определено"
