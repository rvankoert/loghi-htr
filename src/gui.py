# src/gui.py
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import logging
import os
from main import main
from setup.arg_parser import get_arg_parser

class MainApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Model Training GUI")

        # Configure logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.log_text = tk.Text(root, state='disabled', width=80, height=20)
        self.log_text.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

        # Create input fields for each argument
        self.args = {}
        parser = get_arg_parser()
        args = parser.parse_args([])
        args = vars(args)
        num_args = len(args)
        num_columns = 2
        num_rows = (num_args + num_columns - 1) // num_columns
        for i, arg in enumerate(args):
            label = tk.Label(root, text=arg)
            label.grid(row=(i // num_columns) + 1, column=(i % num_columns) * 2, padx=5, pady=5, sticky='w')
            if type(args[arg]) == bool:
                var = tk.BooleanVar(value=args[arg])
                entry = tk.Checkbutton(root, variable=var)
                self.args[arg] = var
            else:
                entry = tk.Entry(root)
                if args[arg] is not None:
                    entry.insert(0, str(args[arg]))
                self.args[arg] = entry
            entry.grid(row=(i // num_columns) + 1, column=(i % num_columns) * 2 + 1, padx=5, pady=5, sticky='w')

        # Create buttons
        self.run_button = tk.Button(root, text="Run", command=self.run_main)
        self.run_button.grid(row=num_rows + 1, column=0, pady=5)

        self.quit_button = tk.Button(root, text="Quit", command=root.quit)
        self.quit_button.grid(row=num_rows + 1, column=1, pady=5)

    def run_main(self):
        self.run_button.config(state='disabled')
        self.log_text.config(state='normal')
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state='disabled')

        # Collect arguments from input fields
        args = {}
        for arg, entry in self.args.items():
            if isinstance(entry, tk.BooleanVar):
                args[arg] = entry.get()
            else:
                args[arg] = entry.get()
        args_list = [f'--{key}={value}' for key, value in args.items() if value]

        # Run the main function in a separate thread
        thread = threading.Thread(target=self.execute_main, args=[args_list])
        thread.start()

    def execute_main(self, args):
        try:
            main(args)
        except Exception as e:
            messagebox.showerror("Error", str(e))
        finally:
            self.run_button.config(state='normal')

    def log_message(self, message):
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, message + '\n')
        self.log_text.config(state='disabled')
        self.log_text.see(tk.END)

def main_gui():
    root = tk.Tk()
    app = MainApp(root)
    root.mainloop()

if __name__ == "__main__":
    main_gui()