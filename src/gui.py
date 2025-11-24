import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
from train import SimpleCNN
from recognize import recognize_image
from process_image import segment_image
from calculate import calculate, save_history
import os

# Настройка
device = torch.device('cpu')  # GUI на ноутбуке лучше на CPU
num_classes = 14
model_path = "../models/handmath_model.pth"

# Проверка наличия модели
if not os.path.exists(model_path):
    messagebox.showerror("Ошибка", f"Файл модели не найден: {model_path}")
    exit()

# Загрузка модели
model = SimpleCNN(num_classes=num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Настройка интерфейса
root = tk.Tk()
root.title("HandMath")

image_path = tk.StringVar()
expression_var = tk.StringVar()
result_var = tk.StringVar()


# Функции
def load_image():
    path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    if path:
        image_path.set(path)
        img = Image.open(path).convert("RGB")
        img.thumbnail((250, 250))
        img_tk = ImageTk.PhotoImage(img)
        label_image.configure(image=img_tk)
        label_image.image = img_tk

# Сегментация
def solve():
    path = image_path.get()
    if not path:
        messagebox.showwarning("Ошибка", "Сначала загрузите изображение")
        return

    symbols = segment_image(path)
    if not symbols:
        messagebox.showwarning("Ошибка", "Не удалось найти символы на изображении")
        return

    expression = ""
    for s in symbols:
        symbol = recognize_image(s, model)
        expression += symbol
    expression_var.set(expression)

    result = calculate(expression)
    result_var.set(result)
    save_history(expression, result)

# История
def show_history():
    history_file = "history/calculations.txt"
    if not os.path.exists(history_file):
        messagebox.showinfo("История", "История пуста")
        return

    with open(history_file, "r", encoding="utf-8") as f:
        history = f.read()

    history_window = tk.Toplevel(root)
    history_window.title("История")
    text = tk.Text(history_window, width=60, height=20)
    text.pack()
    text.insert(tk.END, history)
    text.config(state=tk.DISABLED)


# Виджеты
btn_load = tk.Button(root, text="Загрузить фото", command=load_image)
btn_load.pack(pady=5)

label_image = tk.Label(root)
label_image.pack(pady=5)

btn_solve = tk.Button(root, text="Решение", command=solve)
btn_solve.pack(pady=5)

tk.Label(root, text="Выражение:").pack()
tk.Label(root, textvariable=expression_var, font=("Arial", 14)).pack()

tk.Label(root, text="Результат:").pack()
tk.Label(root, textvariable=result_var, font=("Arial", 14)).pack()

btn_history = tk.Button(root, text="История", command=show_history)
btn_history.pack(pady=5)

root.mainloop()
