import numpy as np
import cv2
import os
import time
import operator
from string import ascii_uppercase
import tkinter as tk
from PIL import Image, ImageTk
from spellchecker import SpellChecker
from keras.models import model_from_json
import threading
from tkinter import ttk
from tkinter import messagebox
import random
from difflib import get_close_matches
from collections import defaultdict
# Application Class
class Application:
    def __init__(self):
        self.spell = SpellChecker()
        self.vs = cv2.VideoCapture(0)
        self.current_image = None
        self.current_image2 = None

        # Language models paths
        self.language_models = {
            'english': ('english.json', 'english.h5'),
            'urdu': ('urdu.json', 'urdu.h5'),
            'german': ('german.json', 'german.h5'),
            'chinese': ('chinese.json', 'chinese.h5'),
        }

        # Load the initial model (default English)
        self.load_model('english')

        # Initialize the character threshold dictionary for English
        self.ct = {char: 0 for char in ascii_uppercase}
        self.ct['blank'] = 0

        # Initialize blank flag
        self.blank_flag = 0

        print("Loaded model from disk")

        # Set up the GUI
        self.setup_gui()

        # Start video loop in a separate thread to avoid blocking the UI
        self.video_thread = threading.Thread(target=self.video_loop)
        self.video_thread.daemon = True
        self.video_thread.start()

        # Initialize empty strings for predictions
        # self.str = ""
        # self.word = ""
        # self.current_symbol = "Empty"
        # self.photo = "Empty"

        self.urdu_dictionary = ["سلام", "محبت", "کتاب", "دوست", "خوشی", "تعلیم", "زندگی", "دوست", "پیار"]
        self.chinese_dictionary = ["你好", "谢谢", "中国", "学习", "朋友", "爱", "书", "快乐"]


    def load_model(self, language):
        """Load the model and character set for the selected language."""
        model_json_file, model_h5_file = self.language_models[language]

        with open(os.path.join("testing_model", model_json_file), "r") as json_file:
            model_json = json_file.read()
        self.loaded_model = model_from_json(model_json)
        self.loaded_model.load_weights(os.path.join("testing_model", model_h5_file))

        model_paths = {
            "dru": ("/Users/muhammadsehelkhan/Desktop/csct/sign_language/Sign-Language-To-Text-Conversion/testing_model/dru.json", "/Users/muhammadsehelkhan/Desktop/csct/sign_language/Sign-Language-To-Text-Conversion/testing_model/dru.h5"),
            "tkdi": ("/Users/muhammadsehelkhan/Desktop/csct/sign_language/Sign-Language-To-Text-Conversion/testing_model/tkdi.json", "/Users/muhammadsehelkhan/Desktop/csct/sign_language/Sign-Language-To-Text-Conversion/testing_model/tkdi.h5"),
            "smn": ("/Users/muhammadsehelkhan/Desktop/csct/sign_language/Sign-Language-To-Text-Conversion/testing_model/smn.json", "/Users/muhammadsehelkhan/Desktop/csct/sign_language/Sign-Language-To-Text-Conversion/testing_model/smn.h5")
        }
        self.models = {}
        for key, (json_path, h5_path) in model_paths.items():
            with open(json_path, "r") as json_file:
                model_json = json_file.read()
            model = model_from_json(model_json)
            model.load_weights(h5_path)
            self.models[key] = model

        # Set the character set based on the language
        if language == 'english':
            self.char_set = ascii_uppercase
        elif language == 'urdu':
            self.char_set = ['ا', 'ب', 'س', 'د', 'ای', 'ف', 'گ', 'ہ', 'آئی', 'جے', 'کے', 'ایل', 'ایم', 'این', 'او', 'پی', 'کیو', 'آر', 'ایس', 'ٹی', 'یو', 'وی', 'ڈبلیو', 'ایکس', 'وائی', 'زی','blank']
        elif language == 'german':
            self.char_set = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'Ä', 'Ö', 'Ü','blank']
        elif language == 'chinese':
            self.char_set = ['阿', '贝', '西', '迪', '伊', '福', '吉', '哈', '艾', '杰', '凯', '艾尔', '艾姆', '恩', '欧', '佩', '奇', '艾尔', '艾س', '提', '尤', '维', '维', '艾克斯', '艾', '兹','blank']



        # Initialize the character count dictionary based on the new character set
        self.ct = {char: 0 for char in self.char_set}
        self.ct['blank'] = 0
        print(f"Loaded {language} model with character set: {self.char_set}")

    def setup_gui(self):
        """Set up the GUI."""
        self.root = tk.Tk()
        self.root.title("Sign Language To Text Conversion")
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)
        self.root.geometry("900x900")
        # self.root.configure(bg="#e8f4f8")  # Soft pastel blue background

        # Set up various labels and panels
        self.panel = tk.Label(self.root)
        self.panel.place(x=100, y=10, width=580, height=580)

        self.panel2 = tk.Label(self.root)
        self.panel2.place(x=400, y=65, width=275, height=275)

        self.T = tk.Label(self.root)
        self.T.place(x=60, y=5)
        self.T.config(text="Sign Language To Text Conversion", font=("Courier", 30, "bold"))

        # Predicted character display
        self.panel3 = tk.Label(self.root)
        self.panel3.place(x=500, y=540)

        self.T1 = tk.Label(self.root)
        self.T1.place(x=10, y=540)
        self.T1.config(text="Character :", font=("Courier", 30, "bold"))

        # Predicted word display
        self.panel4 = tk.Label(self.root)
        self.panel4.place(x=220, y=595)

        self.T2 = tk.Label(self.root)
        self.T2.place(x=10, y=595)
        self.T2.config(text="Word :", font=("Courier", 30, "bold"))


        # Complete sentence display
        self.panel5 = tk.Label(self.root)
        self.panel5.place(x=350, y=645)

        self.T3 = tk.Label(self.root)
        self.T3.place(x=10, y=645)
        self.T3.config(text="Sentence :", font=("Courier", 30, "bold"))

        # Confidence display
        self.confidence_label = tk.Label(self.root)
        self.confidence_label.place(x=700, y=580)

        # Suggestions title
        self.T4 = tk.Label(self.root)
        self.T4.place(x=250, y=690)
        self.T4.config(text="Suggestions :", fg="red", font=("Courier", 30, "bold"))

        # Buttons for suggestions
        self.bt1 = tk.Button(self.root, command=self.action1, height=0, width=0)
        self.bt1.place(x=26, y=745)

        self.bt2 = tk.Button(self.root, command=self.action2, height=0, width=0)
        self.bt2.place(x=325, y=745)

        self.bt3 = tk.Button(self.root, command=self.action3, height=0, width=0)
        self.bt3.place(x=625, y=745)

        self.str = ""
        self.word = " "
        self.current_symbol = "Empty"
        self.photo = "Empty"

        # Language selection dropdown
        self.selected_language = tk.StringVar(value='english')
        languages = ['english', 'urdu', 'german', 'chinese']
        self.language_label = tk.Label(self.root)
        self.language_label.place(x=10, y=850)

        self.language_menu = ttk.OptionMenu(self.root, self.selected_language, languages[0], *languages, command=self.language_selected)
        self.language_menu.place(x=250, y=850)

    def language_selected(self, language):
        """Handle language selection from the dropdown."""
        self.load_model(language)  # Load the selected language model
        self.str = ""  # Reset the sentence
        self.word = ""  # Reset the word
        self.current_symbol = "Empty"
        print(f"Language switched to: {language}")

    def video_loop(self):
        """Process each frame of the video in real-time."""
        while True:
            ok, frame = self.vs.read()

            if ok:
                # Flip and process the video frame
                cv2image = cv2.flip(frame, 1)
                x1 = int(0.5 * frame.shape[1])
                y1 = 10
                x2 = frame.shape[1] - 10
                y2 = int(0.5 * frame.shape[1])

                # Draw rectangle for capturing hand gestures
                cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (255, 0, 0), 1)
                cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGBA)

                # Display the frame in the GUI
                self.current_image = Image.fromarray(cv2image)
                imgtk = ImageTk.PhotoImage(image=self.current_image)
                self.panel.imgtk = imgtk
                self.panel.config(image=imgtk)

                # Crop and process the image for recognition
                cv2image = cv2image[y1:y2, x1:x2]
                gray = cv2.cvtColor(cv2image, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (5, 5), 2)
                th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
                ret, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

                # Predict the character and update the GUI
                self.predict(res)
                self.current_image2 = Image.fromarray(res)
                imgtk = ImageTk.PhotoImage(image=self.current_image2)
                self.panel2.imgtk = imgtk
                self.panel2.config(image=imgtk)

                self.panel3.config(text=self.current_symbol)
                self.panel4.config(text=self.word)
                self.panel5.config(text=self.str)

                # Delay to control frame rate
                time.sleep(0.1)
            else:
                break

    def predict(self, img):
        """Predict the character based on the image input."""
        img = cv2.resize(img, (128, 128))  # Resize to model input size
        img = img.reshape(1, 128, 128, 1)
       
        # Store predictions for the main model and specialized models
        results = {
            "main": self.loaded_model.predict(img),
            "dru": self.models["dru"].predict(img),
            "tkdi": self.models["tkdi"].predict(img),
            "smn": self.models["smn"].predict(img)
        }

        # Mapping main model predictions to characters
        prediction = dict(zip(ascii_uppercase, results["main"][0][1:]))
        prediction['blank'] = results["main"][0][0]

        # Sort predictions by confidence
        sorted_prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
        self.current_symbol = sorted_prediction[0][0]

        # Handle special cases for specific character groups
        if self.current_symbol in 'DRU':
            prediction = {'D': results["dru"][0][0], 'R': results["dru"][0][1], 'U': results["dru"][0][2]}
        elif self.current_symbol == 'T':
            prediction = {'T': results["tkdi"][0][0], 'K': results["tkdi"][0][1], 'D': results["tkdi"][0][2], 'I': results["tkdi"][0][3]}
        elif self.current_symbol == 'S':
            prediction = {'S': results["smn"][0][0], 'M': results["smn"][0][1], 'N': results["smn"][0][2]}
        
        # If applicable, update the current symbol based on specific model predictions
        if self.current_symbol in 'DRUTS':
            sorted_prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
            self.current_symbol = sorted_prediction[0][0]

        # Use the predictions to get the final character
        predictions = self.loaded_model.predict(img)
        char_index = np.argmax(predictions)

        # Validate char_index range
        if char_index < 0 or char_index >= len(self.char_set):
            print(f"Invalid char_index: {char_index}. Skipping character update.")
            return

        char_prediction = self.char_set[char_index]
        confidence = predictions[0][char_index] * 100  # Calculate confidence percentage
        self.confidence_label.config(text=f"Confidence: {confidence:.2f}%")

        # Set threshold for uncertainty
        if confidence < 70:
            self.current_symbol = "Uncertain"
            return

        # Initialize character tracking if needed
        if self.current_symbol not in self.ct:
            self.ct[self.current_symbol] = 0
        self.ct[self.current_symbol] += 1

        # Handle blank predictions and word formation
        if self.current_symbol == 'blank':
            self.blank_flag += 1
            if self.blank_flag > 10 and self.word:
                self.str += self.word + " "
                self.panel5.config(text=self.str)  # Update sentence on GUI
                self.word = ""
                self.blank_flag = 0
        else:
            self.blank_flag = 0
            self.word += char_prediction  # Append character to current word
            self.current_symbol = char_prediction

        # Update GUI with the current word and sentence
        self.panel4.config(text=self.word, font=("Courier", 30))  # Word display
        self.panel5.config(text=self.str, font=("Courier", 30))  # Sentence display

        # Reset character counts after consistent prediction
        if self.ct[self.current_symbol] > 20:
            self.ct = defaultdict(int)  # Reset all counts
            self.str += self.current_symbol
            self.word += self.current_symbol

        # Update the suggestion list based on the detected word
        self.update_suggestions()

    def update_suggestions(self):
        """Update the suggestion list based on the detected word."""
        if self.word:  # Ensure there's a word to provide suggestions for
            suggestions = self.spell.candidates(self.word)  # Get suggestions
            if suggestions:  # Ensure suggestions is not None
                suggestion_list = list(suggestions)[:3]  # Limit to 3 suggestions
                print("Suggestions:", suggestion_list)  # Debug print

                # Update buttons with suggestions
                if len(suggestion_list) > 0:
                    self.bt1.config(text=suggestion_list[0])
                if len(suggestion_list) > 1:
                    self.bt2.config(text=suggestion_list[1])
                if len(suggestion_list) > 2:
                    self.bt3.config(text=suggestion_list[2])
            else:
                self.bt1.config(text="")
                self.bt2.config(text="")
                self.bt3.config(text="")
        else:
            self.bt1.config(text="")
            self.bt2.config(text="")
            self.bt3.config(text="")

    def action1(self):
        """Action for Button 1."""
        self.word = self.bt1.cget("text")
        self.panel4.config(text=self.word)
        print("Suggestion 1 selected")

    def action2(self):
        """Action for Button 2."""
        self.word = self.bt2.cget("text")
        self.panel4.config(text=self.word)
        print("Suggestion 2 selected")

    def action3(self):
        """Action for Button 3."""
        self.word = self.bt3.cget("text")
        self.panel4.config(text=self.word)
        print("Suggestion 3 selected")

    def destructor(self):
        """Clean up resources when closing the application."""
        self.vs.release()
        cv2.destroyAllWindows()
        self.root.quit()

# Run the application
if __name__ == "__main__":
    app = Application()
    app.root.mainloop()


