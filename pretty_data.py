import os
import subprocess
import re
import json5
import json
import random
import numpy as np
import heapq

# Ensure pdflatex is in PATH:
os.environ["PATH"] += r";C:\Users\LATITUDE\AppData\Local\Programs\MiKTeX\miktex\bin\x64"

class Chess:
    def __init__(self, filename):
        with open(filename, encoding='utf-8') as f:
            self.__dict__ = json5.loads(f.read())
            self.board = list(
                (' ' * 9 + '\n') * 2 + ' ' +
                (''.join([c if not c.isdigit() else "." * int(c)
                          for c in self.fen.split()[0].replace('/', '\n ')]) + '\n') +
                (' ' * 9 + '\n') * 2
            )
            self.side = 0 if self.fen.split()[1] == 'w' else 1

    def set_board_as_this_fen(self, this_fen):
        self.board = list(
            (' ' * 9 + '\n') * 2 + ' ' +
            (''.join([c if not c.isdigit() else "." * int(c)
                      for c in this_fen.split()[0].replace('/', '\n ')]) + '\n') +
            (' ' * 9 + '\n') * 2
        )

def fen_to_vector(fen):
    piece_to_value = {
        'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
        'p': -1, 'n': -2, 'b': -3, 'r': -4, 'q': -5, 'k': -6,
        '.': 0
    }
    board_layout = fen.split()[0]
    board_str = ((' ' * 9 + '\n') * 2 +
                 ' ' +
                 ''.join([c if not c.isdigit() else "." * int(c)
                          for c in board_layout.replace('/', '\n ')]) + '\n' +
                 (' ' * 9 + '\n') * 2)
    board_list = list(board_str)
    vector = []
    for row in range(1, 10, 1):
        start = row * 10 + 1
        end = row * 10 + 9
        for idx in range(start, end):
            piece = board_list[idx]
            if piece not in "\n ":
                vector.append(piece_to_value.get(piece, 0))
    return np.array(vector)

def format_vector_as_matrix(vec):
    mat = vec.reshape(8,8)
    lines = []
    for row in mat:
        lines.append(' '.join(f'{int(x):3d}' for x in row))
    return '\n'.join(lines)

def diff_vector_as_matrix(vec1, vec2):
    diff = vec1 - vec2
    return format_vector_as_matrix(diff)

def vector_to_latex(vec, caption=""):
    mat = vec.reshape(8,8)
    latex_str = "\\begin{tabular}{|" + "c|"*8 + "}\n\\hline\n"
    for row in mat:
        row_str = " & ".join(str(int(x)) for x in row) + " \\\\ \\hline\n"
        latex_str += row_str
    latex_str += "\\end{tabular}"
    if caption:
        latex_str = "\\section*{" + caption + "}\n" + latex_str
    return latex_str

def knn_predict(test_vector, training_data, precomputed_norms, k=3):
    """
    Find k-nearest neighbors using precomputed norms for training vectors
    to speed up Euclidean distance calculation.
    """
    test_norm = np.dot(test_vector, test_vector)  # ||test_vector||^2

    distances = [
    (precomputed_norms[i] + test_norm - 2 * np.dot(test_vector, train_vector),
     train_eval, train_fen, test_vector - train_vector)
    for i, (train_vector, train_eval, train_fen) in enumerate(training_data)
    ]

    # Find k smallest distances efficiently using heapq
    top_neighbors = heapq.nsmallest(k, distances, key=lambda x: x[0])

    predicted_eval = sum(score for _, score, _, _ in top_neighbors) / k if k > 0 else 0
    return predicted_eval, top_neighbors

def load_simplified_db(filename):
    data = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            fen = entry.get("fen")
            eval_val = entry.get("eval")
            if fen and eval_val is not None:
                vec = fen_to_vector(fen)
                data.append((vec, eval_val, fen))
    return data

def split_train_test(data, train_ratio):
    random.shuffle(data)
    split_index = int(len(data) * train_ratio)
    return data[:split_index], data[split_index:]

def compile_latex(tex_filename, output_dir="."):
    cmd = ["pdflatex", "-interaction=nonstopmode", tex_filename]
    subprocess.run(cmd, cwd=output_dir)

def main():
    filename = "simplified_db.jsonl"
    data = load_simplified_db(filename)
    print(f"Loaded {len(data)} positions from the simplified database.")
    
    train_data, test_data = split_train_test(data, train_ratio=0.999)
    print(f"Training set: {len(train_data)} positions, Test set: {len(test_data)} positions.")
    
    test_sample_size = max(1, int(len(test_data) * 0.000001))
    test_sample = random.sample(test_data, test_sample_size)
    k = 10
    
    chess = Chess('settings.json')
    
    # We'll accumulate all our output LaTeX in one big string.
    all_output = ("\\documentclass{article}\n"
                  "\\usepackage[utf8]{inputenc}\n"
                  "\\usepackage{booktabs}\n"
                  "\\usepackage{longtable}\n"
                  "\\usepackage{chessboard}\n"  # For rendering chess boards via FEN if desired
                  "\\begin{document}\n")
    precomputed_norms = [np.dot(vec, vec) for vec, _, _ in train_data]
    for idx, (vec, actual_eval, fen) in enumerate(test_sample):
        # In main loop, use precomputed norms:
        predicted_eval, top_neighbors = knn_predict(vec, train_data, precomputed_norms, k=k)
        error = abs(predicted_eval - actual_eval)
        all_output += ("\\section*{Example " + str(idx+1) + "}\n")
        all_output += ("\\textbf{Original Position FEN:} " + fen + "\\\\\n")
        all_output += ("\\textbf{Actual Eval:} " + str(actual_eval) +
                       ", \\textbf{Predicted Eval:} " + f"{predicted_eval:.2f}" +
                       ", \\textbf{Error:} " + f"{error:.2f}" + "\\\\\n")
        
        all_output += ("\\chessboard[setfen=" + fen + "]\\\\\n")

        orig_vector = fen_to_vector(fen)
        all_output += vector_to_latex(orig_vector, caption="Original Board Vector") + "\n"
        
        all_output += "\\subsection*{Top 3 Similar Positions}\n"
        for dist, neigh_eval, neigh_fen, neigh_diff in top_neighbors:
            all_output += ("\\subsubsection*{Neighbor: FEN: " + neigh_fen + "}\n")
            all_output += ("\\textbf{Distance:} " + f"{dist:.2f}" +
                           ", \\textbf{Eval:} " + str(neigh_eval) + "\\\\\n")
            all_output += ("\\chessboard[setfen=" + neigh_fen + "]\\\\\n")
            # Difference vector:
            all_output += vector_to_latex(neigh_diff, caption="Difference Vector") + "\n"
    
    all_output += "\\end{document}\n"
    
    # Write the LaTeX output to a file and compile
    output_dir = "latex_outputs"
    os.makedirs(output_dir, exist_ok=True)
    full_tex_filename = os.path.join(output_dir, "results.tex")
    with open(full_tex_filename, "w", encoding="utf-8") as f_out:
        f_out.write(all_output)
    compile_latex("results.tex", output_dir=output_dir)
    print(f"All output written to {os.path.join(output_dir, 'results.pdf')}.")

if __name__ == "__main__":
    main()
