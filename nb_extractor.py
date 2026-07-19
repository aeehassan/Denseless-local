import json
import os  # Added to handle file paths safely
from pathlib import Path

def extract_headers_and_outputs(notebook_path, output_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
        
    extracted_lines = []
    
    for cell in notebook.get('cells', []):
        cell_type = cell.get('cell_type')
        source = cell.get('source', [])
        
        source_lines = source if isinstance(source, list) else source.splitlines(keepends=True)
        
        if cell_type == 'markdown':
            for line in source_lines:
                clean_line = line.strip()
                if clean_line.startswith('#'):
                    extracted_lines.append(f"\n{clean_line}")
                    
        elif cell_type == 'code':
            outputs = cell.get('outputs', [])
            code_outputs = []
            
            for out in outputs:
                output_type = out.get('output_type')
                
                if output_type in ['stream', 'execute_result']:
                    text_data = out.get('text', out.get('data', {}).get('text/plain', []))
                    text_str = "".join(text_data) if isinstance(text_data, list) else str(text_data)
                    if text_str.strip():
                        code_outputs.append(text_str.strip())
                        
                elif output_type == 'error':
                    traceback = out.get('traceback', [])
                    clean_trace = [line.split('\x1b')[-1] for line in traceback] 
                    code_outputs.append("\n".join(clean_trace))
            
            if code_outputs:
                extracted_lines.append("Code Output:\n" + "\n".join(code_outputs))
                
    # --- FIXED PATH HANDLING ---
    # Automatically get the folder path where the user wants to save the file
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True) # Creates the directory if it's missing
        
    with open(output_path, 'w', encoding='utf-8') as out_file:
        out_file.write("\n".join(extracted_lines))

# --- SAFE PATH USAGE ---
# This ensures it always targets the 'app/agent' directory accurately
script_dir = os.path.dirname(os.path.abspath(__file__))
notebook_file = os.path.join(script_dir, 'app\\agent\\Demo.ipynb')
output_file = os.path.join(script_dir, 'headers_and_outputs.txt')

extract_headers_and_outputs(notebook_file, output_file)
