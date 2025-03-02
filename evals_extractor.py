import json
import io
import zstandard as zstd

def extract_eval_for_turn(position):
    """
    Given a position dictionary, extract the evaluation based on whose turn it is.
    
    - For White (active color 'w'), return the maximum centipawn (cp) value.
    - For Black (active color 'b'), return the minimum cp value.
    
    If no cp values exist, return None.
    """
    fen = position.get("fen", "")
    # The FEN string is assumed to be in the format:
    # "piece_placement active_color castling en_passant"...
    # So, the active color is the second element when splitting on whitespace.
    parts = fen.split()
    if len(parts) < 2:
        # Default to white if something is wrong
        active_color = 'w'
    else:
        active_color = parts[1]
    
    selected_cp = None
    for eval_entry in position.get("evals", []):
        for pv in eval_entry.get("pvs", []):
            cp = pv.get("cp")
            if cp is not None:
                if active_color == 'w':
                    # For white, a higher cp is better.
                    if selected_cp is None or cp > selected_cp:
                        selected_cp = cp
                else:
                    # For black, a lower (more negative) cp is better.
                    if selected_cp is None or cp < selected_cp:
                        selected_cp = cp
    return selected_cp

def process_lichess_file(input_path, output_path, sample_rate):
    dctx = zstd.ZstdDecompressor()
    with open(input_path, 'rb') as compressed_file, open(output_path, 'w', encoding='utf-8') as out_file:
        stream = dctx.stream_reader(compressed_file)
        text_stream = io.TextIOWrapper(stream, encoding='utf-8')
        
        for i, line in enumerate(text_stream):
            if i % sample_rate != 0:
                continue
            position = json.loads(line)
            fen = position.get("fen")
            eval_for_turn = extract_eval_for_turn(position)
            simplified_entry = {"fen": fen, "eval": eval_for_turn}
            out_file.write(json.dumps(simplified_entry) + "\n")

if __name__ == '__main__':
    # Replace these with your actual file paths.
    input_file = "lichess_db_eval.jsonl.zst"
    output_file = "simplified_db.jsonl"
    process_lichess_file(input_file, output_file, sample_rate=100)
