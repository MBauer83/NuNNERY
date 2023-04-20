import tkinter as tk
import re
from tkinter import filedialog
from abc import ABC, abstractmethod
from typing import Callable

class MethodSignature:
    def __init__(self, signature: str, is_static: bool):
        self.signature = signature
        self.is_static = is_static
        self.method_name = signature.split("(")[0]

class InterfaceGenerator:

    def select_target_directory(self):
        # use filedialog to select a directory and set it as the target directory
        self.target_directory_entry.delete(0, tk.END)
        self.target_directory_entry.insert(0, filedialog.askdirectory())


    def __init__(self, root):
        self.method_signature_entry = tk.Entry
        self.root = root
        self.root.title("Interface Generator")

        # Create labels
        tk.Label(root, text="Interface Name:").grid(row=0, column=0, sticky="W")
        tk.Label(root, text="Target Directory:").grid(row=1, column=0, sticky="W")
        tk.Label(root, text="Imports:").grid(row=2, column=3, sticky="W")
        tk.Label(root, text="Method Signatures:").grid(row=2, column=0, sticky="W")

        # Create entry fields
        self.interface_name_entry = tk.Entry(root)
        self.interface_name_entry.grid(row=0, column=1, padx=10, pady=10)
        self.target_directory_entry = tk.Entry(root)
        self.target_directory_entry.grid(row=1, column=1, padx=10, pady=10)

        # use filedialog to select a directory and set it as the target directory
        self.target_directory_button = tk.Button(root, text="Select Target Directory", command=self.select_target_directory)
        self.target_directory_button.grid(row=1, column=2, padx=10, pady=10)

        # Create listbox and buttons for imports
        self.import_listbox = tk.Listbox(root, height=10)
        self.import_listbox.grid(row=2, column=3, columnspan=3, padx=10, pady=10)
        self.import_add_button = tk.Button(root, text="Add Import", command=self.add_import)
        self.import_add_button.grid(row=3, column=3, padx=10, pady=10)
        self.import_remove_button = tk.Button(root, text="Remove Import", command=self.remove_import)
        self.import_remove_button.grid(row=3, column=4, padx=10, pady=10)
        self.import_move_up_button = tk.Button(root, text="Move Import Up", command=self.move_import_up)
        self.import_move_up_button.grid(row=4, column=3, padx=10, pady=10)
        self.import_move_down_button = tk.Button(root, text="Move Import Down", command=self.move_import_down)
        self.import_move_down_button.grid(row=4, column=4, padx=10, pady=10)

        # Create listbox and buttons for method signatures
        self.method_signature_listbox = tk.Listbox(root, height=10)
        self.method_signature_listbox.grid(row=2, column=0, columnspan=3, padx=10, pady=10)
        self.method_sig_add_button = tk.Button(root, text="Add Method Signature", command=self.add_method_signature)
        self.method_sig_add_button.grid(row=3, column=0, padx=10, pady=10)
        self.method_sig_remove_button = tk.Button(root, text="Remove Method Signature", command=self.remove_method_signature)
        self.method_sig_remove_button.grid(row=3, column=1, padx=10, pady=10)
        self.method_sig_move_up_button = tk.Button(root, text="Move Method Signature Up", command=self.move_method_signature_up)
        self.method_sig_move_up_button.grid(row=4, column=0, padx=10, pady=10)
        self.method_sig_move_down_button = tk.Button(root, text="Move Method Signature Down", command=self.move_method_signature_down)
        self.method_sig_move_down_button.grid(row=4, column=1, padx=10, pady=10)

        # Create generate button
        self.generate_button = tk.Button(root, text="Generate Interface", command=self.generate_interface)
        self.generate_button.grid(row=10, column=0, columnspan=3, padx=10, pady=10)

        # Initialize method signatures list
        self.method_signatures: list[MethodSignature] = []

        # Initialize imports list
        self.imports: list[str] = ["from abc import ABC, abstractmethod"]
        self.import_listbox.insert(0, self.imports[0])

    def _handle_button_error(self, fun: Callable[[], None]) -> None:
        try:
            fun()
        except Exception as e:
            tk.messagebox.showerror("Error", str(e))

    def add_import(self):
        # Open a new window to get import
        import_window = tk.Toplevel(self.root)
        import_window.title("Import")

        # make everything twice normal size
        import_window.option_add("*Font", "TkDefaultFont 16")
        import_window.option_add("*Label.Font", "TkDefaultFont 16")

        # Create entry field for import
        import_entry = tk.Entry(import_window)
        import_entry.pack(padx=10, pady=10)

        # Create button to add import to list
        import_add_button = tk.Button(import_window, text="Add Import",
                               command=lambda: self.add_import_callback(import_entry.get(), import_window))
        import_add_button.pack(pady=10)

    def add_import_callback(self, import_name: str, window):
        # check if import is valid
        if not re.match(r"^(?:(?:import)|(?:from)) \w+ .*$", import_name):
            raise Exception("Invalid import name")

        # add import to list
        self.imports.append(import_name)
        self.import_listbox.insert(tk.END, import_name)

        # close window
        window.destroy()

    def remove_import(self):
        # get selected import
        selected_import = self.import_listbox.curselection()
        if len(selected_import) == 0:
            return

        # cannot move or remove the first import "from abc import ABC, abstractmethod"
        if selected_import[0] == 0:
            tk.messagebox.showerror("Error", "Cannot remove the necessary imports for an interface")

        # remove import from list
        self.imports.pop(selected_import[0])
        self.import_listbox.delete(selected_import[0])

    def move_import_up(self):
        # get selected import
        selected_import = self.import_listbox.curselection()
        if len(selected_import) == 0:
            return

        # cannot move or remove the first import "from abc import ABC, abstractmethod"
        if selected_import[0] == 0:
            tk.messagebox.showerror("Error", "Cannot move the necessary imports for an interface")

        # move import up in list
        self.imports.insert(selected_import[0] - 1, self.imports.pop(selected_import[0]))
        self.import_listbox.insert(selected_import[0] - 1, self.import_listbox.get(selected_import[0]))
        self.import_listbox.delete(selected_import[0] + 1)

    def move_import_down(self):
        # get selected import
        selected_import = self.import_listbox.curselection()
        if len(selected_import) == 0:
            return

        # cannot move or remove the first import "from abc import ABC, abstractmethod"
        if selected_import[0] == 0:
            tk.messagebox.showerror("Error", "Cannot move the necessary imports for an interface")

        # move import down in list
        self.imports.insert(selected_import[0] + 1, self.imports.pop(selected_import[0]))
        self.import_listbox.insert(selected_import[0] + 2, self.import_listbox.get(selected_import[0]))
        self.import_listbox.delete(selected_import[0])

    def add_method_signature(self):
        # Open a new window to get method signature
        method_signature_window = tk.Toplevel(self.root)
        method_signature_window.title("Method Signature")

        # Create entry field for method signature
        self.method_signature_entry = tk.Entry(method_signature_window)
        self.method_signature_entry.pack(padx=10, pady=10)

        inner_lambda = lambda :self.add_method_signature_callback(self.method_signature_entry.get(), method_signature_window)
        outer_lambda = lambda :self._handle_button_error(inner_lambda)

        # Create button to add method signature to list
        method_sig_add_button = tk.Button(method_signature_window, text="Add Method Signature", command=outer_lambda)
        method_sig_add_button.pack(pady=10)

    def add_method_signature_callback(self, method_signature, window):
        is_static = re.match(r"\(self,", method_signature)
        # Add method signature to list and update listbox
        self.method_signatures.append(MethodSignature(method_signature, is_static))
        self.method_signature_listbox.delete(0, tk.END)
        for signature in self.method_signatures:
            self.method_signature_listbox.insert(tk.END, signature.signature)
        window.destroy()

    def remove_method_signature(self):
        # Get selected index and delete from list and listbox
        selected_index = self.method_signature_listbox.curselection()
        if selected_index:
            self.method_signatures.pop(selected_index[0])
            self.method_signature_listbox.delete(selected_index)

    def move_method_signature_up(self):
        # Get selected index and move up in list and listbox
        selected_index = self.method_signature_listbox.curselection()
        if selected_index and selected_index[0] > 0:
            self.method_signatures[selected_index[0]], self.method_signatures[selected_index[0] - 1] = \
                self.method_signatures[selected_index[0] - 1], self.method_signatures[selected_index[0]]
            self.method_signature_listbox.delete(0, tk.END)
            for signature in self.method_signatures:
                self.method_signature_listbox.insert(tk.END, signature.signature)
            self.method_signature_listbox.selection_clear(0, tk.END)
            self.method_signature_listbox.selection_set(selected_index[0] - 1)

    def move_method_signature_down(self):
        # Get selected index and move down in list and listbox
        selected_index = self.method_signature_listbox.curselection()
        if selected_index and selected_index[0] < len(self.method_signatures) - 1:
            self.method_signatures[selected_index[0]], self.method_signatures[selected_index[0] + 1] = \
                self.method_signatures[selected_index[0] + 1], self.method_signatures[selected_index[0]]
            self.method_signature_listbox.delete(0, tk.END)
            for signature in self.method_signatures:
                self.method_signature_listbox.insert(tk.END, signature.signature)
            self.method_signature_listbox.selection_clear(0, tk.END)
            self.method_signature_listbox.selection_set(selected_index[0] + 1)

    def generate_interface(self):
        # Get interface name, target directory, and method signatures
        interface_name = self.interface_name_entry.get()
        target_directory = self.target_directory_entry.get()
        method_signatures = self.method_signatures
        print(f'Has {len(method_signatures)} method signatures')

        # Add imports to the top of the file
        class_code = ""
        for import_el in self.imports:
            class_code += import_el + "\n"
        class_code += "\n"

        # Create abstract class with given interface name and method signatures
        class_code += f"class {interface_name}(ABC):\n"
        for signature in method_signatures:
            signature_str = signature.signature
            method_code = ""
            if signature.is_static:
                method_code += "    @staticmethod\n"
            method_code += f"    @abstractmethod\n    def {signature_str}:\n"
            method_code += "        raise NotImplementedError\n"
            class_code += f"\n{method_code}\n"

        # add __subclasshook__ method
        # according to the following example for a class with methods as_array, add, multiply, shape, __len__, and generate
        # @classmethod
        # def __subclasshook__(cls, subclass):
        #     return (
        #         hasattr(subclass, 'as_array') and
        #         callable(subclass.as_array) and
        #         hasattr(subclass, 'add') and
        #         callable(subclass.add) and
        #         hasattr(subclass, 'multiply') and
        #         callable(subclass.multiply) and
        #         hasattr(subclass, 'shape') and
        #         callable(subclass.shape) and
        #         hasattr(subclass, '__len__') and
        #         callable(subclass.__len__) and
        #         hasattr(subclass, 'generate') and
        #         callable(subclass.generate) or
        #         NotImplemented
        #     )
        class_code += f"    @classmethod\n    def __subclasshook__(cls, subclass):\n"
        class_code += "        return (\n"
        last_method = method_signatures[-1]
        # iterate over methods 1..n-1 to end with "and\n" instead of "or\n"
        # then add the last method with "or\n" instead of "and\n" and finally,
        # add "NotImplemented" and the closing parentheses.
        for method in method_signatures[:-1]:
            class_code += f"            (hasattr(subclass, '{method.method_name}') and\n"
            class_code += f"            callable(subclass.{method.method_name}) and\n"
        class_code += f"            hasattr(subclass, '{last_method.method_name}') and\n"
        class_code += f"            callable(subclass.{last_method.method_name})) or\n"
        class_code += "            NotImplemented\n"
        class_code += "        )\n"


        # Save interface to file in target directory
        with open(f"{target_directory}/{interface_name}.py", "w") as f:
            f.write(class_code)

        # Display success message
        tk.messagebox.showinfo("Interface Generated", f"Successfully generated {interface_name} interface in {target_directory}")
        
root = tk.Tk()
interface_generator = InterfaceGenerator(root)
root.mainloop()