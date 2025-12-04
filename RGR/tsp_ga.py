import numpy as np
import random as py_random
import matplotlib.pyplot as plt
import io
import os
import threading
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import traceback
import time
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image


# 1. Вспомогательные функции

def distance(city1, city2):
    return np.linalg.norm(city1 - city2)

def route_length(route, cities):
    total = 0.0
    n = len(route)
    for i in range(n):
        total += distance(cities[route[i]], cities[route[(i + 1) % n]])
    return total

def fitness(route, cities):
    return 1.0 / (1e-6 + route_length(route, cities))

def init_population(pop_size, n_cities):
    population = []
    base = list(range(n_cities))
    for _ in range(pop_size):
        ind = base[:]
        py_random.shuffle(ind)
        population.append(ind)
    return population


# 2. Генетические операторы

def selection(population, cities, k=3):
    selected = []
    for _ in range(len(population)):
        candidates = py_random.sample(population, k)
        best = max(candidates, key=lambda ind: fitness(ind, cities))
        selected.append(best[:])
    return selected

def pmx_crossover(parent1, parent2):
    n = len(parent1)
    child = [-1] * n
    a, b = sorted(py_random.sample(range(n), 2))
    child[a:b] = parent1[a:b]
    mapping = {}
    for i in range(a, b):
        mapping[parent2[i]] = parent1[i]
    for i in range(n):
        if child[i] == -1:
            val = parent2[i]
            visited = set()
            while val in child:
                if val in visited:
                    break
                visited.add(val)
                if val in mapping:
                    val = mapping[val]
                else:
                    break
            if val in child:
                available = set(range(n)) - set(child)
                if available:
                    val = available.pop()
                else:
                    val = i
            child[i] = val
    assert len(set(child)) == n
    return child

def ox_crossover(parent1, parent2):
    size = len(parent1)
    child = [-1] * size
    start, end = sorted(py_random.sample(range(size), 2))
    child[start:end] = parent1[start:end]
    ptr = end
    for city in parent2[end:] + parent2[:end]:
        if city not in child:
            while child[ptr] != -1:
                ptr = (ptr + 1) % size
            child[ptr] = city
    return child

def cx_crossover(parent1, parent2):
    size = len(parent1)
    child = [-1] * size
    index = 0
    while child[index] == -1:
        child[index] = parent1[index]
        index = parent1.index(parent2[index])
    for i in range(size):
        if child[i] == -1:
            child[i] = parent2[i]
    return child

def mutate(individual, mutation_rate=0.02):
    for i in range(len(individual)):
        if py_random.random() < mutation_rate:
            j = py_random.randint(0, len(individual) - 1)
            individual[i], individual[j] = individual[j], individual[i]


# 3. Генетический алгоритм

def genetic_algorithm_with_history(
    cities,
    pop_size=80,
    generations=100,
    mutation_rate=0.03,
    crossover_func=pmx_crossover,
    elite_size=2,
    log_callback=None
):
    n_cities = len(cities)
    population = init_population(pop_size, n_cities)
    best_per_gen = []
    bar_width = 30

    if log_callback:
        log_callback("=" * 60)
        log_callback(f"🚀 ЗАПУСК ГА (TSP)")
        log_callback(f" Городов: {n_cities} | Популяция: {pop_size}")
        log_callback(f" Поколений: {generations} | Мутация: {mutation_rate:.2%}")
        log_callback(f" Элита: {elite_size} | Кроссовер: {crossover_func.__name__}")
        log_callback("-" * 60)
        log_callback(f"{'Пок':>4} | {'Лучший':>8} | {'Средний':>8} | Прогресс")

    for gen in range(generations):
        lengths = [route_length(ind, cities) for ind in population]
        best_len = min(lengths)
        avg_len = np.mean(lengths)
        best_route = population[np.argmin(lengths)]
        best_per_gen.append(best_route[:])

        if log_callback:
            progress = (gen + 1) / generations
            bars = "█" * int(progress * bar_width)
            spaces = " " * (bar_width - len(bars))
            bar_str = f"[{bars}{spaces}] {int(progress * 100):>3}%"
            log_callback(f"{gen+1:>4} | {best_len:>8.3f} | {avg_len:>8.3f} | {bar_str}")

        elite_indices = np.argsort(lengths)[:elite_size]
        elite = [population[i][:] for i in elite_indices]

        selected = selection(population, cities)
        next_gen = elite[:]
        while len(next_gen) < pop_size:
            p1, p2 = py_random.sample(selected, 2)
            child1 = crossover_func(p1[:], p2[:])
            child2 = crossover_func(p2[:], p1[:])
            mutate(child1, mutation_rate)
            mutate(child2, mutation_rate)
            next_gen.extend([child1, child2])
        population = next_gen[:pop_size]

    if log_callback:
        log_callback("-" * 60)
        log_callback(f"✅ ЗАВЕРШЕНО. Лучший маршрут: {best_len:.4f}")
        log_callback("=" * 60)
    
    return best_per_gen


# 4. Визуализация: анимация в tkinter

def animate_in_tkinter(cities, best_history, parent_window, save_gif=False, gif_path="tsp_evolution.gif", fps=30):
    anim_win = tk.Toplevel(parent_window)
    anim_win.title("Анимация эволюции TSP")
    anim_win.geometry("650x680")

    fig = Figure(figsize=(6, 6), dpi=100)
    ax = fig.add_subplot(111)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.5)

    ax.scatter(cities[:, 0], cities[:, 1], c='red', s=100, zorder=5)
    for i, (x, y) in enumerate(cities):
        ax.text(x + 0.01, y + 0.01, str(i), fontsize=10, color='darkred', fontweight='bold')

    line, = ax.plot([], [], 'b-o', lw=2, markersize=7, markerfacecolor='lightblue', markeredgecolor='navy')

    canvas = FigureCanvasTkAgg(fig, master=anim_win)
    canvas.get_tk_widget().pack(fill="both", expand=True)

    control_frame = ttk.Frame(anim_win)
    control_frame.pack(pady=5)

    is_paused = tk.BooleanVar(value=False)
    def toggle_pause():
        is_paused.set(not is_paused.get())
        btn_pause.config(text="▶ Продолжить" if is_paused.get() else "⏸ Пауза")

    btn_pause = ttk.Button(control_frame, text="⏸ Пауза", command=toggle_pause)
    btn_pause.pack(side="left", padx=5)

    btn_close = ttk.Button(control_frame, text="❌ Закрыть", command=anim_win.destroy)
    btn_close.pack(side="left", padx=5)

    status_label = ttk.Label(control_frame, text="Поколение: 0 / 0", font=("Arial", 10))
    status_label.pack(side="left", padx=10)

    frames = []
    delay_ms = int(1000 / fps)

    def update_frame(frame_idx=0):
        if frame_idx >= len(best_history):
            status_label.config(text="✅ Анимация завершена", foreground="green")
            return

        if is_paused.get():
            anim_win.after(delay_ms, update_frame, frame_idx)
            return

        route = best_history[frame_idx]
        coords = np.array([cities[i] for i in route] + [cities[route[0]]])
        line.set_data(coords[:, 0], coords[:, 1])
        ax.set_title(
            f"TSP: Эволюция маршрута\nПоколение {frame_idx+1}/{len(best_history)} | Длина: {route_length(route, cities):.3f}",
            fontsize=12
        )
        canvas.draw()

        status_label.config(text=f"Поколение: {frame_idx+1} / {len(best_history)}")

        if save_gif:
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            frames.append(Image.open(buf).convert("P", palette="ADAPTIVE"))

        anim_win.after(delay_ms, update_frame, frame_idx + 1)

    update_frame()

    def on_close():
        if save_gif and frames:
            try:
                folder = os.path.dirname(gif_path)
                if folder and not os.path.exists(folder):
                    os.makedirs(folder)
                frames[0].save(
                    gif_path,
                    save_all=True,
                    append_images=frames[1:],
                    duration=delay_ms,
                    loop=0,
                    optimize=True
                )
                messagebox.showinfo("✅ Успешно", f"GIF сохранён:\n{gif_path}")
            except Exception as e:
                messagebox.showerror("❌ Ошибка", f"Не удалось сохранить GIF:\n{e}")
        anim_win.destroy()

    anim_win.protocol("WM_DELETE_WINDOW", on_close)

def plot_convergence(histories, labels, title="Сравнение операторов кроссовера"):
    plt.figure(figsize=(10, 6))
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    for i, (hist, label) in enumerate(zip(histories, labels)):
        color = colors[i % len(colors)]
        plt.plot(hist['best_lengths'], label=f"Лучший ({label})", color=color)
        plt.plot(hist['avg'], '--', alpha=0.7, label=f"Средний ({label})", color=color)
    plt.xlabel("Поколение")
    plt.ylabel("Длина маршрута")
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

def animate_single_route(cities, route, length, parent_window, save_gif=False, gif_path="tsp_evolution.gif", fps=60):
    anim_win = tk.Toplevel(parent_window)
    anim_win.title("Анимация лучшего маршрута")
    anim_win.geometry("650x680")

    fig = Figure(figsize=(6, 6), dpi=100)
    ax = fig.add_subplot(111)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.5)

    ax.scatter(cities[:, 0], cities[:, 1], c='red', s=100, zorder=5)
    for i, (x, y) in enumerate(cities):
        ax.text(x + 0.01, y + 0.01, str(i), fontsize=10, color='darkred', fontweight='bold')

    coords = np.array([cities[i] for i in route] + [cities[route[0]]])
    line, = ax.plot(coords[:, 0], coords[:, 1], 'b-o', lw=2, markersize=7, markerfacecolor='lightblue', markeredgecolor='navy')

    canvas = FigureCanvasTkAgg(fig, master=anim_win)
    canvas.get_tk_widget().pack(fill="both", expand=True)

    control_frame = ttk.Frame(anim_win)
    control_frame.pack(pady=5)

    btn_close = ttk.Button(control_frame, text="❌ Закрыть", command=anim_win.destroy)
    btn_close.pack(side="left", padx=5)

    ax.set_title(f"Лучший маршрут\nДлина: {length:.3f}", fontsize=12)
    canvas.draw()

    # Для сохранения GIF — рисуем один кадр
    frames = []
    if save_gif:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        frames.append(Image.open(buf).convert("P", palette="ADAPTIVE"))

        try:
            folder = os.path.dirname(gif_path)
            if folder and not os.path.exists(folder):
                os.makedirs(folder)
            frames[0].save(
                gif_path,
                save_all=True,
                append_images=frames[1:],
                duration=1000//fps,
                loop=0,
                optimize=True
            )
            messagebox.showinfo("✅ Успешно", f"GIF сохранён:\n{gif_path}")
        except Exception as e:
            messagebox.showerror("❌ Ошибка", f"Не удалось сохранить GIF:\n{e}")

    anim_win.protocol("WM_DELETE_WINDOW", anim_win.destroy)

# 5. Графический интерфейс

crossover_func_global = "PMX"

class TSPApp:
    def __init__(self, root):
        self.root = root
        self.root.title("РГЗ: Генетический алгоритм для TSP")
        self.root.geometry("900x720")
        self.root.resizable(True, True)

        self.cities = None
        self.best_history = []
        self.histories = []
        self.labels = []

        self.setup_ui()

    def setup_ui(self):
        param_frame = ttk.LabelFrame(self.root, text="Параметры задачи", padding=10)
        param_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(param_frame, text="Городов:").grid(row=0, column=0, sticky="w")
        self.n_cities_var = tk.IntVar(value=20)
        ttk.Spinbox(param_frame, from_=3, to=20, textvariable=self.n_cities_var, width=5).grid(row=0, column=1, padx=5)

        ttk.Label(param_frame, text="Seed:").grid(row=0, column=2, sticky="w", padx=(15,5))
        self.seed_var = tk.StringVar(value="50")
        ttk.Entry(param_frame, textvariable=self.seed_var, width=24).grid(row=0, column=3, padx=5)
        ttk.Label(param_frame, text="(пусто = случайный)").grid(row=0, column=4, sticky="w", padx=(5,0))

        self.input_mode = tk.StringVar(value="auto")
        ttk.Radiobutton(param_frame, text="Случайная генерация", variable=self.input_mode, value="auto").grid(row=1, column=0, columnspan=2, sticky="w", pady=(5,0))
        ttk.Radiobutton(param_frame, text="Ввести вручную", variable=self.input_mode, value="manual").grid(row=1, column=2, columnspan=2, sticky="w", pady=(5,0))
        self.btn_coords = ttk.Button(param_frame, text="📝 Ввести координаты", command=self.open_coords_window)
        self.btn_coords.grid(row=1, column=4, padx=(10,0), pady=(5,0))
        self.btn_coords.config(state="disabled")
        self.input_mode.trace("w", lambda *a: self.btn_coords.config(state="normal" if self.input_mode.get()=="manual" else "disabled"))

        ga_frame = ttk.LabelFrame(self.root, text="Параметры ГА", padding=10)
        ga_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(ga_frame, text="Популяция:").grid(row=0, column=0, sticky="w")
        self.pop_size_var = tk.IntVar(value=70)
        ttk.Spinbox(ga_frame, from_=10, to=200, textvariable=self.pop_size_var, width=6).grid(row=0, column=1, padx=5)

        ttk.Label(ga_frame, text="Поколения:").grid(row=0, column=2, sticky="w", padx=(15,0))
        self.generations_var = tk.IntVar(value=100)
        ttk.Spinbox(ga_frame, from_=10, to=500, textvariable=self.generations_var, width=6).grid(row=0, column=3, padx=5)

        ttk.Label(ga_frame, text="Мутация:").grid(row=1, column=0, sticky="w", pady=5)
        self.mutation_var = tk.DoubleVar(value=0.04)
        ttk.Spinbox(ga_frame, from_=0.0, to=1.0, increment=0.01, textvariable=self.mutation_var, width=6).grid(row=1, column=1, padx=5)

        ttk.Label(ga_frame, text="Элита:").grid(row=1, column=2, sticky="w", padx=(15,0), pady=5)
        self.elite_var = tk.IntVar(value=2)
        ttk.Spinbox(ga_frame, from_=0, to=10, textvariable=self.elite_var, width=6).grid(row=1, column=3, padx=5)

        crossover_frame = ttk.LabelFrame(self.root, text="Оператор кроссовера", padding=10)
        crossover_frame.pack(fill="x", padx=10, pady=5)
        self.cross_var = tk.StringVar(value="all")
        for i, (text, val) in enumerate([("PMX", "pmx"), ("OX", "ox"), ("CX", "cx"), ("Все три", "all")]):
            ttk.Radiobutton(crossover_frame, text=text, variable=self.cross_var, value=val).pack(side="left", padx=8)

        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(pady=10)
        self.btn_run = ttk.Button(btn_frame, text="🚀 Запустить ГА", command=self.run_ga)
        self.btn_run.pack(side="left", padx=5)
        self.btn_animate = ttk.Button(btn_frame, text="🎬 Анимация", command=self.show_animation, state="disabled")
        self.btn_animate.pack(side="left", padx=5)
        self.btn_save_gif = ttk.Button(btn_frame, text="💾 Сохранить GIF", command=self.save_gif, state="disabled")
        self.btn_save_gif.pack(side="left", padx=5)
        self.btn_show_plot = ttk.Button(btn_frame, text="📈 График", command=self.show_convergence_plot, state="disabled")
        self.btn_show_plot.pack(side="left", padx=5)

        self.progress = ttk.Progressbar(self.root, mode='determinate')
        self.progress.pack(fill="x", padx=10, pady=5)

        log_frame = ttk.LabelFrame(self.root, text="Лог выполнения", padding=5)
        log_frame.pack(fill="both", expand=True, padx=10, pady=5)
        self.log_text = scrolledtext.ScrolledText(log_frame, height=12, font=("Consolas", 9))
        self.log_text.pack(fill="both", expand=True)
        self.log_text.config(state="disabled")

    def get_seed(self):
        s = self.seed_var.get().strip()
        return int(s) if s else None

    def set_seed(self, seed):
        if seed is not None:
            np.random.seed(seed)
            py_random.seed(seed)
        else:
            import time
            seed = int(time.time() * 100000) % (2**32 - 1)
            np.random.seed(seed)
            py_random.seed(seed)
        return seed

    def clear_log(self):
        self.log_text.config(state="normal")
        self.log_text.delete(1.0, "end")
        self.log_text.config(state="disabled")

    def open_coords_window(self):
        n = self.n_cities_var.get()
        win = tk.Toplevel(self.root)
        win.title(f"Введите координаты {n} городов")
        win.geometry("400x500")

        canvas = tk.Canvas(win)
        scrollbar = ttk.Scrollbar(win, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        entries = []
        for i in range(n):
            ttk.Label(scrollable_frame, text=f"Город {i}:").grid(row=i, column=0, padx=5, pady=2, sticky="w")
            x_ent = ttk.Entry(scrollable_frame, width=8)
            y_ent = ttk.Entry(scrollable_frame, width=8)
            x_ent.grid(row=i, column=1, padx=2)
            y_ent.grid(row=i, column=2, padx=2)
            entries.append((x_ent, y_ent))

        def on_ok():
            try:
                coords = []
                for i, (x_ent, y_ent) in enumerate(entries):
                    x = float(x_ent.get().strip())
                    y = float(y_ent.get().strip())
                    coords.append([x, y])
                self.cities = np.array(coords)
                win.destroy()
                self.log(f"✅ Координаты заданы вручную для {n} городов.")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Неверный формат:\n{e}")

        ttk.Button(scrollable_frame, text="OK", command=on_ok).grid(row=n, column=0, columnspan=3, pady=10)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def log(self, msg):
        if self.root:
            self.root.after(0, lambda: self._log_threadsafe(msg))

    def _log_threadsafe(self, msg):
        self.log_text.config(state="normal")
        self.log_text.insert("end", str(msg) + "\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def run_ga(self):
        self.clear_log()
        try:
            seed = self.get_seed()

            if self.input_mode.get() == "manual":
                if self.cities is None:
                    messagebox.showwarning("Внимание", "Сначала введите координаты!")
                    return
                cities = self.cities
                self.log(f"✅ Используем ручные координаты ({len(cities)} городов).")
            else:
                if seed is None:
                    seed = int(time.time() * 100000) % (2**32 - 1)
                    self.log(f"🌱 Использован случайный seed: {seed}")
                else:
                    self.log(f"🌱 Использован seed: {seed}")
                
                seed = self.set_seed(seed)
                n = self.n_cities_var.get()
                cities = np.random.rand(n, 2)
                self.cities = cities
                self.log(f"✅ Сгенерировано {n} городов.")

            cross_map = {
                "pmx": ([pmx_crossover], ["PMX"]),
                "ox":  ([ox_crossover],  ["OX"]),
                "cx":  ([cx_crossover],  ["CX"]),
                "all": ([pmx_crossover, ox_crossover, cx_crossover], ["PMX", "OX", "CX"])
            }
            funcs, labels = cross_map[self.cross_var.get()]

            for btn in [self.btn_run, self.btn_animate, self.btn_save_gif, self.btn_show_plot]:
                btn.config(state="disabled")

            self.histories = []
            self.labels = labels

            def ga_worker():
                try:
                    self.set_seed(seed)
                    for func, label in zip(funcs, labels):
                        global crossover_func_global
                        crossover_func_global = label
                        self.log(f"\n▶ Запуск ГА ({label})...")

                        best_hist = genetic_algorithm_with_history(
                            cities=cities,
                            pop_size=self.pop_size_var.get(),
                            generations=self.generations_var.get(),
                            mutation_rate=self.mutation_var.get(),
                            crossover_func=func,
                            elite_size=self.elite_var.get(),
                            log_callback=self.log
                        )

                        best_lengths = [route_length(r, cities) for r in best_hist]
                        avg_lengths = [np.mean(best_lengths[max(0, i-4):i+1]) for i in range(len(best_lengths))]

                        self.histories.append({'best_lengths': best_lengths, 'best_routes': best_hist, 'avg': avg_lengths})

                        if func == funcs[-1]:
                            self.best_history = best_hist


                    self.root.after(0, lambda: self.on_ga_complete(cities, labels, seed))
                except Exception as e:
                    err_msg = str(e)
                    tb_msg = traceback.format_exc()
                    self.root.after(0, lambda em=err_msg, tm=tb_msg: 
                        messagebox.showerror("❌ Ошибка", f"{em}\n\n{tm}"))

            threading.Thread(target=ga_worker, daemon=True).start()

        except Exception as e:
            messagebox.showerror("❌ Ошибка", str(e))


    def on_ga_complete(self, cities, labels, seed):
        # Разблокируем кнопки в начале
        for btn in [self.btn_run, self.btn_animate, self.btn_save_gif]:
            btn.config(state="normal")
        if len(self.histories) > 1:
            self.btn_show_plot.config(state="normal")

        best_overall = min(min(h['best_lengths']) for h in self.histories)
        seed_display = seed if seed is not None else "случайный"
        self.log(f"\n✅ ГА завершён. Seed: {seed_display}")
        self.log(f"🏆 Лучшая длина: {best_overall:.4f}")

        # === НОВОЕ: находим историю с лучшим маршрутом ===
        best_route_global = None
        best_hist_for_animation = None
        for hist in self.histories:
            lengths = hist['best_lengths']
            routes = hist['best_routes']
            min_idx = np.argmin(lengths)
            route = routes[min_idx]
            if best_route_global is None or route_length(route, cities) < route_length(best_route_global, cities):
                best_route_global = route
                best_hist_for_animation = routes  # вся история поколений для анимации

        self.best_overall_route = best_route_global  # для одиночного отображения
        self.best_history_for_animation = best_hist_for_animation  # для анимации эволюции

        epsilon = 0.01

        if len(self.histories) > 1:
            self.log("\n" + "="*50)
            self.log("📈 Сравнение скорости сходимости (ε = 1%):")
            self.log("Кроссовер | Поколение достижения цели")
            self.log("-" * 42)
            for i, (hist, label) in enumerate(zip(self.histories, self.labels)):
                G_eps = self.calculate_convergence_generation(hist['best_lengths'], epsilon)
                self.log(f"{label:<9} | {G_eps:>25}")
            self.log("="*50)
        else:
            self.log(f"\n📊 Скорость сходимости (ε = {epsilon:.0%}):")
            for i, (hist, label) in enumerate(zip(self.histories, self.labels)):
                G_eps = self.calculate_convergence_generation(hist['best_lengths'], epsilon)
                self.log(f"  • {label}: {G_eps} поколений")
            

    def show_animation(self):
        if not hasattr(self, 'best_history_for_animation') or self.cities is None:
            messagebox.showwarning("Внимание", "Сначала запустите ГА!")
            return
        try:
            animate_in_tkinter(
                cities=self.cities,
                best_history=self.best_history_for_animation,
                parent_window=self.root,
                save_gif=False,
                fps=60
            )
        except Exception as e:
            messagebox.showerror("❌ Ошибка анимации", str(e))

            
    def save_gif(self):
        if not self.best_history or self.cities is None:
            messagebox.showwarning("Внимание", "Сначала запустите ГА!")
            return

        file_path = filedialog.asksaveasfilename(
            title="Сохранить GIF",
            defaultextension=".gif",
            filetypes=[("GIF files", "*.gif")],
            initialfile="tsp_evolution.gif"
        )
        if not file_path:
            return

        try:
            animate_in_tkinter(
                cities=self.cities,
                best_history=self.best_history,
                parent_window=self.root,
                save_gif=True,
                gif_path=file_path,
                fps=60
            )
        except Exception as e:
            messagebox.showerror("❌ Ошибка", f"Не удалось сохранить GIF:\n{e}")

    def show_convergence_plot(self):
        if not self.histories:
            return
        plot_convergence(self.histories, self.labels)

    def calculate_convergence_generation(self, best_lengths, epsilon=0.01):
        if not best_lengths:
            return None
        
        L_star = min(best_lengths)
        L_target = L_star * (1 + epsilon)

        for gen, L in enumerate(best_lengths):
            if L <= L_target:
                return gen + 1
        return len(best_lengths)


# 6. Запуск приложения

if __name__ == "__main__":
    root = tk.Tk()
    app = TSPApp(root)
    root.mainloop()