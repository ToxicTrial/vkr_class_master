import os
import cv2
import time
import tkinter as tk
from tkinter import messagebox, filedialog, ttk
from PIL import Image, ImageTk
import qrcode
import threading
import func
import fp
from save_face import capture_faces
from embedding_create import get_embeddings

class App:
    def __init__(self, root, device, detector, inception, embeddings):
        self.root = root
        self.root.title("Система распознавания")
        
        base_dir = os.path.dirname(os.getcwd())
        self.database = os.path.join(base_dir, "database")
        self.fprints = os.path.join(self.database, "fingerprints")
        self.embeddings_path = os.path.join(self.database, "embeddings")
        self.cards_path = os.path.join(self.database, "cards")
        self.faces_path = os.path.join(self.database, "faces")

        self.embeddings = embeddings
        self.device = device
        self.detector = detector
        self.inception = inception
        self.fingerprints = fp.Fingerprints()
        
        self.running = True
        self.current_mode = None
        self.current_user = None
        
        self.show_main_menu()

    def show_main_menu(self):
        """Главное меню с выбором режима"""
        self.clear_window()
        self.current_mode = None
        
        title = tk.Label(self.root, text="Система распознавания", font=("Arial", 16))
        title.pack(pady=20)
        
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=20)
        
        ttk.Button(btn_frame, text="Добавить пользователя", 
                 command=self.start_add_user_mode).pack(pady=10, ipadx=20, ipady=5)
        
        ttk.Button(btn_frame, text="Распознать пользователя", 
                 command=self.start_recognition_mode).pack(pady=10, ipadx=20, ipady=5)
        
        ttk.Button(btn_frame, text="Выход", 
                 command=self.on_close).pack(pady=10, ipadx=20, ipady=5)

    def start_add_user_mode(self):
        """Режим добавления пользователя"""
        self.current_mode = 'add_user'
        self.clear_window()
        
        tk.Label(self.root, text="Введите имя пользователя:").pack(pady=10)
        self.user_name_entry = ttk.Entry(self.root)
        self.user_name_entry.pack(pady=5)

        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=20)
        
        ttk.Button(btn_frame, text="Сделать фото лица", 
                 command=self.capture_user_face).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(btn_frame, text="Добавить отпечаток", 
                 command=self.add_fingerprint).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(btn_frame, text="Создать карту", 
                 command=self.create_user_card).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(self.root, text="Назад", 
                 command=self.show_main_menu).pack(pady=20)

    def start_recognition_mode(self):
        """Режим распознавания пользователя"""
        self.current_mode = 'recognize'
        self.clear_window()
        
        self.video_label = tk.Label(self.root)
        self.video_label.pack()
        
        ttk.Button(self.root, text="Назад", 
                 command=self.show_main_menu).pack(pady=20)
        
        self.cap = func.camera_connect(0)
        if not self.cap.isOpened():
            messagebox.showerror("Ошибка", "Не удалось открыть камеру")
            self.show_main_menu()
            return
        
        self.start_time = time.time()
        self.card_reader()

    def capture_user_face(self):
        username = self.user_name_entry.get().strip()
        if not username:
            messagebox.showerror("Ошибка", "Введите имя пользователя")
            return
        
        # Захват изображений лица
        success = capture_faces(username)
        
        if success:
            # Обновление эмбеддинга
            from embedding_create import update_user_embedding
            if update_user_embedding(username):
                messagebox.showinfo("Успех", "Изображения и эмбеддинг успешно сохранены")
            else:
                messagebox.showwarning("Предупреждение", "Изображения сохранены, но не удалось обновить эмбеддинг")
        else:
            messagebox.showerror("Ошибка", "Не удалось сохранить изображения")

    def add_fingerprint(self):
        """Добавление отпечатка пальца"""
        username = self.user_name_entry.get().strip()
        if not username:
            messagebox.showerror("Ошибка", "Введите имя пользователя")
            return
            
        user_fprints_path = os.path.join(self.fprints, username)
        os.makedirs(user_fprints_path, exist_ok=True)
        
        file_path = filedialog.askopenfilename(
            title="Выберите изображение отпечатка пальца",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")]
        )
        
        if file_path:
            try:
                dest_path = os.path.join(user_fprints_path, "f1.jpg")
                img = Image.open(file_path)
                img.save(dest_path)
                messagebox.showinfo("Успех", "Отпечаток пальца успешно добавлен")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось сохранить отпечаток: {str(e)}")

    def create_user_card(self):
        """Создание QR-кода карты пользователя"""
        username = self.user_name_entry.get().strip()
        if not username:
            messagebox.showerror("Ошибка", "Введите имя пользователя")
            return
            
        # Создаем папку для карт пользователя
        user_cards_path = os.path.join(self.cards_path, username)
        os.makedirs(user_cards_path, exist_ok=True)

        # Генерируем QR-код
        try:
            # Создаем объект QR-кода
            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_L,
                box_size=10,
                border=4,
            )
            
            # Добавляем данные (используем имя пользователя как идентификатор)
            qr.add_data(username)
            qr.make(fit=True)
            
            # Создаем изображение QR-кода
            img = qr.make_image(fill_color="black", back_color="white")
            
            # Сохраняем QR-код в файл
            card_path = os.path.join(user_cards_path, f"{username}_card.png")
            img.save(card_path)
            
            # Показываем пользователю сгенерированный QR-код
            self.show_qr_code(card_path)
            
            messagebox.showinfo("Успех", f"QR-код карты для {username} успешно создан")
            self.show_main_menu()
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось создать QR-код: {str(e)}")

    def show_qr_code(self, image_path):
        """Показывает QR-код в отдельном окне"""
        qr_window = tk.Toplevel(self.root)
        qr_window.title("Ваш QR-код")
        
        # Загружаем изображение
        img = Image.open(image_path)
        img = ImageTk.PhotoImage(img)
        
        # Создаем и размещаем элементы интерфейса
        label = tk.Label(qr_window, image=img)
        label.image = img  # сохраняем ссылку на изображение
        label.pack(padx=10, pady=10)
        
        # Кнопка закрытия
        close_btn = ttk.Button(qr_window, text="Закрыть", 
                            command=qr_window.destroy)
        close_btn.pack(pady=5)

    def card_reader(self):
        """Чтение карты и распознавание лица"""
        def process_frame():
            if not self.running or self.current_mode != 'recognize':
                return
                
            ret, frame = self.cap.read()
            if not ret:
                self.root.after(100, process_frame)
                return

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
            
            if time.time() - self.start_time > 1.0: 
                self.start_time = time.time()
                detected, card = self.read_card(frame)
                if detected:
                    self.process_detected_card(card)
            
            self.root.after(30, process_frame)
        
        process_frame()

    def read_card(self, frame):
        """Распознавание QR-кода"""
        detector = cv2.QRCodeDetector()
        data, _, _ = detector.detectAndDecode(frame)
        return (True, data) if data else (False, None)

    def process_detected_card(self, card):
        """Обработка обнаруженной карты"""
        card_file = os.path.join(self.cards_path, card, f"{card}_card.png")
        if os.path.isfile(card_file):
            self.current_user = card
            messagebox.showinfo("Успех", f"Карта определена")
            self.verify_user()
        else:
            messagebox.showerror("Ошибка", f"Карта не соответствует требованиям")

    def verify_user(self):
        """Проверка пользователя по лицу и отпечатку"""
        ret, frame = self.cap.read()
        if not ret:
            return
            
        faces = func.detect_faces(frame, self.embeddings, self.detector, self.device, self.inception)
        print(faces)
        face_verified = any(face['name'] == self.current_user and face['score'] > 0.8 for face in faces)
        
        if face_verified:
            fingerprint_path = os.path.join(self.fprints, self.current_user, "f1.jpg")
            if os.path.exists(fingerprint_path):
                file_path = filedialog.askopenfilename(
                    title="Выберите изображение отпечатка для проверки",
                    filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")]
                )
                
                if file_path:
                    score = self.fingerprints.compare_images_cosine(file_path, fingerprint_path)
                    if score > 0.8:
                        messagebox.showinfo("Успех", f"Пользователь {self.current_user} идентифицирован!")
                    else:
                        messagebox.showerror("Ошибка", "Несоответствие отпечатков")
            else:
                messagebox.showerror("Ошибка", "Отпечаток не найден в базе")
        else:
            messagebox.showerror("Ошибка", f"Лицо не соответствует пользователю {self.current_user}")
        
        self.current_user = None

    def clear_window(self):
        """Очистка окна"""
        for widget in self.root.winfo_children():
            widget.destroy()
        
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()

    def on_close(self):
        """Завершение работы"""
        self.running = False
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        self.root.destroy()

class LoadingScreen:
    def __init__(self):
        self.root = tk.Tk()
        self.root.withdraw()

        self.top = tk.Toplevel(self.root)
        self.top.title("Загрузка...")
        self.top.geometry("300x100")
        self.top.resizable(False, False)
        self.label = tk.Label(self.top, text="Загрузка моделей, подождите...", font=("Arial", 12))
        self.label.pack(expand=True, padx=20, pady=30)
        self.top.protocol("WM_DELETE_WINDOW", lambda: None)
        self.top.grab_set()

    def destroy(self):
        self.root.destroy()

class LoadingScreen:
    def __init__(self):
        self.root = tk.Tk()
        self.root.withdraw()

        self.top = tk.Toplevel(self.root)
        self.top.title("Загрузка...")
        self.top.geometry("300x100")
        self.top.resizable(False, False)
        self.label = tk.Label(self.top, text="Загрузка моделей, подождите...", font=("Arial", 12))
        self.label.pack(expand=True, padx=20, pady=30)
        self.top.protocol("WM_DELETE_WINDOW", lambda: None)
        self.top.grab_set()

    def destroy(self):
        self.top.destroy()
        self.root.destroy()

if __name__ == "__main__":
    loading = LoadingScreen()

    app_data = {}

    def background_load():
        device, detector, inception = func.load_models()
        embeddings_path = os.path.join(os.getcwd(), "database", "embeddings")
        embeddings = func.load_embeddings(embeddings_path)
        app_data["device"] = device
        app_data["detector"] = detector
        app_data["inception"] = inception
        app_data["embeddings"] = embeddings
        loading.root.after(0, start_main_app)

    def start_main_app():
        loading.destroy()
        root = tk.Tk()
        app = App(root,
                  app_data["device"],
                  app_data["detector"],
                  app_data["inception"],
                  app_data["embeddings"])
        root.protocol("WM_DELETE_WINDOW", app.on_close)
        root.mainloop()

    threading.Thread(target=background_load, daemon=True).start()
    loading.root.mainloop()
