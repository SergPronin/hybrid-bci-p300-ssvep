import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import re
from scipy.linalg import eigh


def load_eeg(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    header_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "FP1" in stripped.upper() or "FPZ" in stripped.upper():
            header_idx = i
            break

    if header_idx is None:
        raise ValueError("Не найдена строка с каналами")

    header_line = lines[header_idx].strip().replace(",", " ")
    channels = [ch.strip() for ch in header_line.split() if ch.strip()]
    data_lines = lines[header_idx + 1:]

    number_pattern = re.compile(r"[-+]?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?")
    data = []
    for line in data_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        nums = number_pattern.findall(stripped)
        if len(nums) < len(channels):
            continue
        try:
            row = [float(x.replace(",", ".")) for x in nums[:len(channels)]]
            data.append(row)
        except ValueError:
            continue

    data = np.array(data)
    if data.size == 0:
        raise ValueError("Не удалось считать числовые данные из файла")
    df = pd.DataFrame(data, columns=channels)
    return df


def compute_cca(df, group1, group2, n_components=1):
    X = df[group1].values.astype(np.float64)
    Y = df[group2].values.astype(np.float64)

    x_std = X.std(axis=0, ddof=1)
    y_std = Y.std(axis=0, ddof=1)

    zero_x = [group1[i] for i, v in enumerate(x_std) if v <= 1e-15 or not np.isfinite(v)]
    zero_y = [group2[i] for i, v in enumerate(y_std) if v <= 1e-15 or not np.isfinite(v)]
    if zero_x or zero_y:
        details = []
        if zero_x:
            details.append("Группа 1 (нулевая дисперсия): " + ", ".join(zero_x))
        if zero_y:
            details.append("Группа 2 (нулевая дисперсия): " + ", ".join(zero_y))
        raise ValueError("Нельзя выполнить CCA: найдены константные каналы.\n" + "\n".join(details))

    X = (X - X.mean(axis=0)) / x_std
    Y = (Y - Y.mean(axis=0)) / y_std

    if not np.isfinite(X).all() or not np.isfinite(Y).all():
        raise ValueError("В выбранных каналах есть NaN/Inf после стандартизации")

    n = X.shape[0]
    p = X.shape[1]
    q = Y.shape[1]

    n_components = min(n_components, p, q)

    Sxx = (X.T @ X) / (n - 1)
    Syy = (Y.T @ Y) / (n - 1)
    Sxy = (X.T @ Y) / (n - 1)

    Syy_inv = np.linalg.pinv(Syy)

    Cxx = Sxy @ Syy_inv @ Sxy.T
    eigvals, eigvecs = eigh(Cxx, Sxx)

    idx = np.argsort(-eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    correlations = np.sqrt(np.clip(eigvals[:n_components], 0.0, None))

    A = eigvecs[:, :n_components].astype(np.float64)
    B = np.zeros((q, n_components), dtype=np.float64)

    for i in range(n_components):
        if correlations[i] < 1e-15:
            continue
        B[:, i] = (Syy_inv @ Sxy.T @ A[:, i]) / correlations[i]

    for i in range(n_components):
        norm_a = np.sqrt(max(A[:, i].T @ Sxx @ A[:, i], 1e-15))
        norm_b = np.sqrt(max(B[:, i].T @ Syy @ B[:, i], 1e-15))
        A[:, i] /= norm_a
        B[:, i] /= norm_b

    for i in range(n_components):
        U = X @ A[:, i]
        V = Y @ B[:, i]

        load_x = np.array([np.corrcoef(X[:, j], U)[0, 1] for j in range(p)])
        load_y = np.array([np.corrcoef(Y[:, j], V)[0, 1] for j in range(q)])

        idx_max_x = np.argmax(np.abs(load_x))
        idx_max_y = np.argmax(np.abs(load_y))

        sign_x = np.sign(load_x[idx_max_x])
        sign_y = np.sign(load_y[idx_max_y])

        sign = sign_x if abs(load_x[idx_max_x]) >= abs(load_y[idx_max_y]) else sign_y

        if sign < 0:
            A[:, i] *= -1
            B[:, i] *= -1

    return correlations, A, B


class EEGCCAApp:
    def __init__(self, master):
        self.master = master
        master.title("EEG CCA Analyzer")
        master.geometry('900x600')
        master.configure(bg='#2e2e2e')

        self.df = None
        self.channels = []
        self.group1_vars = []
        self.group2_vars = []
        self.group1_checkboxes = []
        self.group2_checkboxes = []
        self.select_all1_var = tk.BooleanVar()
        self.select_all2_var = tk.BooleanVar()

        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TButton', font=('Arial', 11), padding=5,
                        background='#444', foreground='white')
        style.map('TButton', background=[('active', '#666')])
        style.configure('TLabel', font=('Arial', 11), padding=3,
                        background='#2e2e2e', foreground='white')
        style.configure('TEntry', fieldbackground='#444', foreground='white')

        self.load_button = ttk.Button(master, text="Загрузить EEG файл",
                                     command=self.load_file)
        self.load_button.pack(pady=5)

        self.top_frame = tk.Frame(master, bg='#2e2e2e')
        self.top_frame.pack(fill='both', expand=True, padx=10, pady=5)

        self.channel_frame = tk.Frame(self.top_frame, bg='#2e2e2e')
        self.channel_frame.grid(row=0, column=0, sticky='nw')

        self.result_frame = tk.Frame(self.top_frame, bg='#2e2e2e')
        self.result_frame.grid(row=0, column=1, sticky='ns', padx=(10, 0))

        self.group1_label = ttk.Label(self.channel_frame, text="Группа 1")
        self.group1_label.grid(row=0, column=0, sticky='w')

        self.select_all1_cb = tk.Checkbutton(
            self.channel_frame, text="Выбрать все",
            variable=self.select_all1_var,
            command=self.toggle_select_all1,
            bg='#2e2e2e', fg='white', selectcolor='#444'
        )
        self.select_all1_cb.grid(row=1, column=0, sticky='w')

        self.group2_label = ttk.Label(self.channel_frame, text="Группа 2")
        self.group2_label.grid(row=0, column=1, sticky='w')

        self.select_all2_cb = tk.Checkbutton(
            self.channel_frame, text="Выбрать все",
            variable=self.select_all2_var,
            command=self.toggle_select_all2,
            bg='#2e2e2e', fg='white', selectcolor='#444'
        )
        self.select_all2_cb.grid(row=1, column=1, sticky='w')

        self.group1_box = tk.Frame(self.channel_frame, bg='#2e2e2e')
        self.group1_box.grid(row=2, column=0, sticky='nw', padx=10)

        self.group2_box = tk.Frame(self.channel_frame, bg='#2e2e2e')
        self.group2_box.grid(row=2, column=1, sticky='nw', padx=10)

        self.ncomp_label = ttk.Label(master, text="Количество компонент CCA")
        self.ncomp_label.pack(pady=5)

        self.ncomp_entry = ttk.Entry(master)
        self.ncomp_entry.insert(0, '1')
        self.ncomp_entry.pack(pady=5)

        self.run_button = ttk.Button(master, text="Выполнить CCA",
                                    command=self.run_cca)
        self.run_button.pack(pady=5)

        self.result_label = ttk.Label(self.result_frame, text="Результаты:")
        self.result_label.pack(anchor='nw')

        self.result_text = tk.Text(self.result_frame,
                                   font=('Arial', 10),
                                   bg='#1e1e1e',
                                   fg='white',
                                   width=40)
        self.result_text.pack(fill='both', expand=True)
        self.result_text.configure(state='disabled')

        self.top_frame.grid_columnconfigure(0, weight=1)
        self.top_frame.grid_columnconfigure(1, weight=1)
        self.top_frame.grid_rowconfigure(0, weight=1)

    def load_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Text files", "*.txt")],
            title="Выберите EEG файл"
        )
        if not file_path:
            return

        try:
            self.df = load_eeg(file_path)
            self.channels = list(self.df.columns)
            self.update_channel_checkboxes()

            messagebox.showinfo(
                "Успех",
                f"Файл загружен\nКоличество каналов: {len(self.channels)}"
            )
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def update_channel_checkboxes(self):
        for widget in self.group1_box.winfo_children():
            widget.destroy()
        for widget in self.group2_box.winfo_children():
            widget.destroy()

        self.group1_vars = []
        self.group2_vars = []
        self.group1_checkboxes = []
        self.group2_checkboxes = []

        for ch in self.channels:
            var = tk.BooleanVar()
            var.trace_add('write', lambda *args: self.update_select_all_state('group1'))
            cb = tk.Checkbutton(self.group1_box, text=ch, variable=var,
                               bg='#2e2e2e', fg='white',
                               selectcolor='#444')
            cb.pack(anchor='w')
            self.group1_vars.append(var)
            self.group1_checkboxes.append(cb)

        for ch in self.channels:
            var = tk.BooleanVar()
            var.trace_add('write', lambda *args: self.update_select_all_state('group2'))
            cb = tk.Checkbutton(self.group2_box, text=ch, variable=var,
                               bg='#2e2e2e', fg='white',
                               selectcolor='#444')
            cb.pack(anchor='w')
            self.group2_vars.append(var)
            self.group2_checkboxes.append(cb)

    def update_select_all_state(self, group):
        if group == 'group1':
            vars_list = self.group1_vars
            select_all_var = self.select_all1_var
        else:
            vars_list = self.group2_vars
            select_all_var = self.select_all2_var

        if vars_list:
            all_selected = all(var.get() for var in vars_list)
            select_all_var.set(all_selected)

    def toggle_select_all1(self):
        state = self.select_all1_var.get()
        for var in self.group1_vars:
            var.set(state)

    def toggle_select_all2(self):
        state = self.select_all2_var.get()
        for var in self.group2_vars:
            var.set(state)

    def run_cca(self):
        if self.df is None:
            messagebox.showwarning("Внимание", "Сначала загрузите EEG файл")
            return

        group1 = [self.channels[i] for i, v in enumerate(self.group1_vars) if v.get()]
        group2 = [self.channels[i] for i, v in enumerate(self.group2_vars) if v.get()]

        if not group1 or not group2:
            messagebox.showwarning("Внимание", "Выберите каналы в обеих группах")
            return

        try:
            n_components = int(self.ncomp_entry.get())
            correlations, x_weights, y_weights = compute_cca(
                self.df, group1, group2, n_components
            )
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))
            return

        self.result_text.configure(state='normal')
        self.result_text.delete('1.0', tk.END)

        self.result_text.insert(tk.END, "Канонические корреляции:\n\n")
        for i, corr in enumerate(correlations):
            self.result_text.insert(tk.END, f"Компонента {i+1}: {corr:.6f}\n")

        self.result_text.insert(tk.END, "\nКанонические веса:\n")

        for i in range(len(correlations)):
            self.result_text.insert(tk.END, f"\nКомпонента {i+1}:\n")

            self.result_text.insert(tk.END, "  Группа 1:\n")
            for ch, w in zip(group1, x_weights[:, i]):
                self.result_text.insert(tk.END, f"    {ch}: {w:.6f}\n")

            self.result_text.insert(tk.END, "  Группа 2:\n")
            for ch, w in zip(group2, y_weights[:, i]):
                self.result_text.insert(tk.END, f"    {ch}: {w:.6f}\n")

        self.result_text.configure(state='disabled')


if __name__ == "__main__":
    root = tk.Tk()
    app = EEGCCAApp(root)
    root.mainloop()