import tkinter as tk
from tkinter import messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from tkinter import filedialog
import functions
from start_program import get_data
from start_program import detect_objects


class SpectralPanelApp:
    def __init__(self, root):
        self.root = root
        self.language = 'r'
        self.original_rect_coords = None
        self.warning_shown = False  # Флаг для отслеживания, было ли предупреждение
        self.highlighted_all_rectangles = False
        self.add_mode = tk.BooleanVar(value=True)  # True — добавление, False — редактирование

        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.rectangles = []
        self.rect_coords = []
        self.dragging = False
        self.editing = False
        self.edit_rect_index = None
        self.edit_corner = None
        self.selected_index = None
        self.start_x = None
        self.start_y = None
        self.current_rect = None

        # Окно выбора файла
        self.select_file_button = tk.Button(self.root, text="Select File", command=self.select_file)
        self.select_file_button.pack(side=tk.TOP, fill=tk.X)

        self.frame = tk.Frame(self.root)
        self.frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('motion_notify_event', self.on_drag)
        self.canvas.mpl_connect('button_release_event', self.on_release)

        self.frame = tk.Frame(self.root)
        self.frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.mode_checkbox = tk.Checkbutton(
            self.frame,
            text="Add Rectangles Mode",
            variable=self.add_mode,
            onvalue=True,
            offvalue=False,
            command = self.toggle_mode)
        self.mode_checkbox.pack(side=tk.TOP)

        self.delete_button = tk.Button(self.frame, text="Delete Selected", command=self.delete_selected_rectangle)
        self.delete_button.pack(side=tk.LEFT)

        self.deselect_button = tk.Button(self.frame, text="Deselect", command=self.deselect_rectangle)
        self.deselect_button.pack(side=tk.LEFT)

        self.select_all_button = tk.Button(self.frame, text="Highlight  All Rectangles", command=self.highlight_all_rectangles)
        self.select_all_button.pack(side=tk.LEFT)

        self.language_button = tk.Button(self.frame, text="Язык: Русский", command=self.toggle_language)
        self.language_button.pack(side=tk.RIGHT)

        self.save_button = tk.Button(self.frame, text="Save XlSX-report", command=self.save_report)
        self.save_button.pack(side=tk.RIGHT)

        self.save_las_button = tk.Button(self.frame, text="Save Las-report", command=self.save_las_report)
        self.save_las_button.pack(side=tk.RIGHT)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def select_file(self):
        file_path = filedialog.askopenfilename(title="Select a File",
                                               filetypes=(("Las files", "*.las"), ("All files", "*.*")))
        if file_path:
            self.load_file(file_path)

    def toggle_mode(self):
        if self.add_mode.get():
            self.mode_checkbox.config(text="Add Rectangles Mode")
        else:
            self.mode_checkbox.config(text="Edit Rectangles Mode")

    def load_file(self, file_path):
        self.frequencies, self.depth, self.aps_data = get_data(file_path)
        # После загрузки данных обновляем панель
        self.rectangles.clear()
        self.rect_coords.clear()
        self.ax.clear()
        self.visualise_aps_panel()
        self.rectangles_data = detect_objects(self.frequencies, self.depth, self.aps_data)
        if self.rectangles_data is not None:
            self.draw_rectangles_from_data(self.rectangles_data)
        self.canvas.draw()

    def toggle_language(self):
        if self.language == 'r':
            self.language = 'e'
            self.language_button.config(text="Language: English")
        else:
            self.language = 'r'
            self.language_button.config(text="Язык: Русский")

    def visualise_aps_panel(self):
        self.ax.clear()
        vmin_color = np.percentile(self.aps_data, 85)
        vmax_color = np.percentile(self.aps_data, 99)
        colors = ['white', 'blue', 'cyan', 'green', 'yellow', 'orange', 'red', 'maroon']
        cmap = LinearSegmentedColormap.from_list('custom_colormap', colors)
        self.ax.pcolormesh(self.frequencies, self.depth, self.aps_data,
                           cmap=cmap, vmin=vmin_color, vmax=vmax_color, shading='auto')
        self.ax.set_xlabel('Frequency, kHz')
        self.ax.set_ylabel('Depth, m')
        self.ax.set_title('Spectral Panel')
        self.ax.invert_yaxis()
        self.canvas.draw()


    def draw_rectangles_from_data(self, rectangles_data):
        for coords in rectangles_data:
            x_min, y_min, x_max, y_max = coords
            # Проверка на допустимость координат
            if not self.is_within_bounds(x_min, y_min, x_max, y_max):
                continue
            # Проверка на пересечение
            new_rect = {
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max
            }
            if self.check_overlap(new_rect):
                continue

            # Добавляем прямоугольник
            bottom = min(y_min, y_max)
            height = abs(y_max - y_min)
            rect = Rectangle(
                (x_min, bottom),
                x_max - x_min,
                height,
                linewidth=1,
                edgecolor='maroon',
                facecolor='red',
                alpha=0.5,
            )
            self.ax.add_patch(rect)
            self.rectangles.append(rect)
            self.rect_coords.append(new_rect)
        self.canvas.draw()

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        x = event.xdata
        y = event.ydata
        if not self.add_mode.get():
            # Сначала проверка: попали ли внутрь прямоугольника
            for idx, coords in enumerate(self.rect_coords):
                x_min, x_max = sorted([coords['x_min'], coords['x_max']])
                y_min, y_max = sorted([coords['y_min'], coords['y_max']])

                if x_min <= x <= x_max and y_min <= y <= y_max:
                    self.select_rectangle(idx)
                    # Найдём ближайший угол
                    corners = [
                        (x_min, y_min),  # top-left
                        (x_max, y_min),  # top-right
                        (x_max, y_max),  # bottom-right
                        (x_min, y_max)  # bottom-left
                    ]
                    dists = [(i, (x - cx) ** 2 + (y - cy) ** 2) for i, (cx, cy) in enumerate(corners)]
                    closest_corner = min(dists, key=lambda item: item[1])[0]

                    self.editing = True
                    self.edit_rect_index = idx
                    self.edit_corner = closest_corner
                    self.original_rect_coords = self.rect_coords[idx].copy()
                    return
            return

        # Если режим добавления
        self.start_x, self.start_y = float(x), float(y)
        self.current_rect = Rectangle(
            (x, y), 0, 0,
            linewidth=1,
            edgecolor='black',
            facecolor='red',
            alpha=0.5
        )
        self.ax.add_patch(self.current_rect)
        self.dragging = True
        self.canvas.draw()


    def on_drag(self, event):
        if event.inaxes != self.ax or  self.warning_shown or (not self.dragging and not self.editing):
            return
        x = event.xdata
        y = event.ydata
        if self.dragging and self.current_rect:
            x0 = min(self.start_x, x)
            y0 = min(self.start_y, y)
            width = abs(x - self.start_x)
            height = abs(y - self.start_y)
            self.current_rect.set_xy((x0, y0))
            self.current_rect.set_width(width)
            self.current_rect.set_height(height)
            self.canvas.draw()
        elif self.editing:
            rect = self.rectangles[self.edit_rect_index]
            coords = self.rect_coords[self.edit_rect_index]

            # Копируем координаты
            x1, x2 = coords['x_min'], coords['x_max']
            y1, y2 = coords['y_min'], coords['y_max']
            if self.edit_corner == 0:  # top-left
                x1, y1 = x, y
            elif self.edit_corner == 1:  # top-right
                x2, y1 = x, y
            elif self.edit_corner == 2:  # bottom-right
                x2, y2 = x, y
            elif self.edit_corner == 3:  # bottom-left
                x1, y2 = x, y

            # Обновление координат
            coords['x_min'], coords['x_max'] = sorted([x1, x2])
            coords['y_min'], coords['y_max'] = sorted([y1, y2])
            rect.set_xy((coords['x_min'], coords['y_min']))
            rect.set_width(coords['x_max'] - coords['x_min'])
            rect.set_height(coords['y_max'] - coords['y_min'])
            self.canvas.draw()

    def on_release(self, event):
        if event.inaxes != self.ax:
            return
        x_end = event.xdata
        y_end = event.ydata
        if self.dragging and self.current_rect:
            x0 = float(min(self.start_x, x_end))
            y0 = float(min(self.start_y, y_end))
            x1 = float(max(self.start_x, x_end))
            y1 = float(max(self.start_y, y_end))

            new_rect = {
                "x_min": x0,
                "y_min": y0,
                "x_max": x1,
                "y_max": y1
            }
            # Проверка на перекрытие с другими прямоугольниками
            if self.check_overlap(new_rect):
                self.warning_shown = True  # Устанавливаем флаг
                messagebox.showwarning("Overlap", "Rectangles cannot overlap.")
                self.current_rect.remove()
                self.current_rect = None
                self.dragging = False
                self.canvas.draw()
                self.warning_shown = False  # Сбрасываем флаг после закрытия окна
                return
            self.rectangles.append(self.current_rect)
            self.rect_coords.append(new_rect)
            self.current_rect = None
            self.dragging = False
            self.canvas.draw()
        elif self.editing:
            index = self.edit_rect_index
            coords = self.rect_coords[index]
            rect = self.rectangles[index]  # rect = прямоугольник с нужным индексом

            new_rect = {
                "x_min": coords["x_min"],
                "y_min": coords["y_min"],
                "x_max": coords["x_max"],
                "y_max": coords["y_max"]
            }
            # Временно удаляем текущий, чтобы не проверять его с самим собой
            temp_coords = self.rect_coords[:index] + self.rect_coords[index+1:]

            if self.check_overlap(new_rect, temp_coords):
                self.warning_shown = True  # Устанавливаем флаг
                messagebox.showwarning("Overlap", "Rectangles cannot overlap.")
                # Откат к оригинальным координатам
                coords['x_min'] = self.original_rect_coords['x_min']
                coords['x_max'] = self.original_rect_coords['x_max']
                coords['y_min'] = self.original_rect_coords['y_min']
                coords['y_max'] = self.original_rect_coords['y_max']
                rect.set_xy((coords['x_min'], coords['y_min']))
                rect.set_width(coords['x_max'] - coords['x_min'])
                rect.set_height(coords['y_max'] - coords['y_min'])
                self.canvas.draw()
            else:
                # Если пересечения нет, обновляем координаты и перерисовываем
                coords['x_min'], coords['x_max'] = sorted([new_rect['x_min'], new_rect['x_max']])
                coords['y_min'], coords['y_max'] = sorted([new_rect['y_min'], new_rect['y_max']])
                rect.set_xy((coords['x_min'], coords['y_min']))
                rect.set_width(coords['x_max'] - coords['x_min'])
                rect.set_height(coords['y_max'] - coords['y_min'])
                self.canvas.draw()
            self.editing = False
            self.edit_rect_index = None
            self.edit_corner = None
            self.warning_shown = False


    def check_overlap(self, new_rect, other_rects=None):
        if other_rects is None:
            other_rects = self.rect_coords
        for rect in other_rects:
            if not (
                    new_rect['x_max'] <= rect['x_min'] or
                    new_rect['x_min'] >= rect['x_max'] or
                    new_rect['y_max'] <= rect['y_min'] or
                    new_rect['y_min'] >= rect['y_max']
            ):
                return True  # Есть пересечение
        return False


    def is_within_bounds(self, x_min, y_min, x_max, y_max):
        x_range = (self.frequencies[0], self.frequencies[len(self.frequencies)-1])
        y_range = (self.depth[0], self.depth[len(self.depth)-1])
        return (x_range[0] <= x_min <= x_range[1] and
                x_range[0] <= x_max <= x_range[1] and
                y_range[0] <= y_min <= y_range[1] and
                y_range[0] <= y_max <= y_range[1])


    def select_rectangle(self, index):
        if self.selected_index is not None:
            self.rectangles[self.selected_index].set_facecolor('red')
        self.selected_index = index
        self.rectangles[index].set_facecolor('green')
        self.canvas.draw()


    def deselect_rectangle(self):
        if self.selected_index is not None:
            self.rectangles[self.selected_index].set_facecolor('red')
            self.selected_index = None
            self.canvas.draw()


    def delete_selected_rectangle(self):
        if self.selected_index is not None:
            self.rectangles[self.selected_index].remove()
            del self.rectangles[self.selected_index]
            del self.rect_coords[self.selected_index]
            self.selected_index = None
            self.canvas.draw()


    def save_report(self):
        # Предложить пользователю выбрать путь для сохранения Excel-файла
        file_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")],
            title="Сохранить отчёт как"
        )
        if file_path:
            try:
                functions.upload_report_to_excel(
                    self.rect_coords,
                    self.frequencies,
                    self.depth,
                    self.aps_data,
                    self.language,
                    file_path  # передаём путь для сохранения
                )
                messagebox.showinfo("Успех", "Отчёт успешно сохранён!")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось сохранить отчёт:\n{e}")


    def save_las_report(self):
        # Предложить пользователю выбрать путь для сохранения las-файла
        file_path = filedialog.asksaveasfilename(
            defaultextension=".las",
            filetypes=[("Las files", "*.las")],
            title="Сохранить отчёт как"
        )
        if file_path:
            try:
                functions.upload_intervals_to_las_file(
                    self.rect_coords,
                    self.depth,
                    file_path  # передаём путь для сохранения
                )
                messagebox.showinfo("Успех", "Отчёт успешно сохранён!")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось сохранить отчёт:\n{e}")


    def highlight_all_rectangles(self):
        # Пройдем по всем прямоугольникам и изменим их цвет
        if(self.highlighted_all_rectangles):
            color = 'red'
            alpha = 0.5
        else:
            color = 'yellow'
            alpha = 0.9
        for rect in self.rectangles:
            rect.set_facecolor(color)
            rect.set_alpha(alpha)
        self.canvas.draw()
        self.highlighted_all_rectangles = not self.highlighted_all_rectangles


    def on_close(self):
        self.root.quit()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Spectral Panel with selected objects")
    app = SpectralPanelApp(root)
    root.mainloop()

