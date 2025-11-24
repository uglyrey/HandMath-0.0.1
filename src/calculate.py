import os
from datetime import datetime

def calculate(expression):
    try:
        result = eval(expression)
        return result
    except Exception:
        return "Ошибка"

def save_history(expression, result):
    if not os.path.exists("history"):
        os.makedirs("history")
    with open('history/calculations.txt', 'a', encoding='utf-8') as f:
        f.write(f"{datetime.now()} | {expression} = {result}\n")
