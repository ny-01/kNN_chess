import json5
import json
import random
import numpy as np

class Chess:
    def __init__(self, filename):
        with open(filename, encoding='utf-8') as f:
            # now that we are constructing the actual chess object,
            # we load up the settings json file
            # and immediately set a bunch of fields that we will need in the chess object
            # e.g. chess.directions, all of which comes from this json file
            self.__dict__ = json5.loads(f.read())
            # we produce the initial position of the board by using the fen string in the dictionary
            # (' ' * 9 + '\n') we have this at the beginning and end of the list, representing a padding of 2 ranks above and below the board
            # (''.join([c if not c.isdigit() else "." * int(c) for c in self.fen.split()[0].replace('/', '\n ')]) is for the actual board itself
            # the reason we say c if not c.isdigit() is because if c is a digit then it needs a separate case- i.e. if it's 8 then we have an empty rank
            # i.e. "." 8 times (. represents an empty square)
            # otherwise if c isn't a digit then we simply represent c
            self.board = list((' ' * 9 + '\n') * 2 + ' '\
                         + (''.join([c if not c.isdigit() else "." * int(c) for c in self.fen.split()[0].replace('/', '\n ')])+'\n')\
                         + (' ' * 9 + '\n') * 2)
            # The board indices are structured as follows:
            # Each line ends with a newline character '\n' and starts with a space ' ':
            # 0  1  2  3  4  5  6  7  8  9 - padding
            # 10 11 12 13 14 15 16 17 18 19 - padding
            # 20 21 22 23 24 25 26 27 28 29 - black's pieces
            # 30 31 32 33 34 35 36 37 38 39 - black's pawns
            # 40 41 42 43 44 45 46 47 48 49 - empty 3rd rank
            # 50 51 52 53 54 55 56 57 58 59 - empty 4th rank
            # 60 61 62 63 64 65 66 67 68 69 - empty 5th rank
            # 70 71 72 73 74 75 76 77 78 79 - empty 6th rank
            # 80 81 82 83 84 85 86 87 88 89 - white's pieces
            # 90 91 92 93 94 95 96 97 98 99 - white's pawns
            # 100 101 102 103 104 105 106 107 108 109 - padding
            # 110 111 112 113 114 115 116 117 118 119 - padding

            # NOTE: when printing out the board, since all positions in the list ending 9
            # (i.e. the right side of the board) are newline chars, when you print the board
            # the number of visible chars will be 120-12=108

            # the padding is actually REALLY useful for calculating what "off the board" is
            self.side = 0 if self.fen.split()[1] == 'w' else 1
    
    def set_board_as_this_fen(self, this_fen):
        self.board = list((' ' * 9 + '\n') * 2 + ' '\
                     + (''.join([c if not c.isdigit() else "." * int(c) for c in this_fen.split()[0].replace('/', '\n ')])+'\n')\
                     + (' ' * 9 + '\n') * 2)

    # All TODO statements in the below method are stuff that you can comment in to visualise the moves being generated
    # all this function does is generates the legal moves for the current position
    def generate_moves(self):
        move_list=[]
        for square in range(len(self.board)):
            # the length of self.board is 120- 10 a row (which comes from the space plus pieces plus newline),
            # 12 times because there are 8 ranks on a chess board plus 2 on top and bottom for padding
            piece = self.board[square] # accessing the elements of the board one by one
            if piece not in ' .\n' and self.colors[piece] == self.side:
                # if it is indeed a piece and of the side of that whose turn it is move,
                for offset in self.directions[piece]:
                    # we will print each possible direction possible for that piece along with the piece itself
                    # TODO: print(piece, offset)
                    # set target_square to initially be the square which the piece is on
                    # we're going to use target_square to determine if the piece can actually move to that square
                    # and whether we should actually stop trying one particular direction that a piece can move in and explore the next direction
                    # e.g. for a bishop this would mean we hit into a friendly piece (discard), or we've hit into an enemy piece (accept)
                    # and then moving on as a result
                    target_square = square
                    while True:
                        # we're going to iteratively move the piece
                        target_square += offset
                        # captured_piece represents what is actually at the new position that we are moving our piece to
                        captured_piece = self.board[target_square]
                        # if we hit a square that is out of the board range then stop- do not continue and move onto the next available offset
                        if captured_piece in ' \n': break
                        # if we haven't broken yet, then we're on the board- there are 3 options now
                        # the square we're on is either an empty space, one of our own pieces, or a piece of the opposite colour
                        
                        # if it's a piece of our colour, then don't move our piece onto that square and don't print the result
                        if self.colors[captured_piece] == self.side: break
                        if piece in 'Pp':
                            # if we're moving a pawn on either side and the offset we're considering is that of a diagonal (i.e. capturing)
                            # then we shouldn't change the board and print anything unless we really are capturing something
                            if offset in [9, 11, -9, -11] and captured_piece == '.': break
                            # if we're moving a pawn forward and there's a piece in the way, do not proceed (pawns don't capture this way)
                            # other pieces don't have this restriction of capturing in a particular way- they capture when they move to a square
                            if offset in [10, 20, -10, -20] and captured_piece != '.': break
                        # should we make this move? if we're considering a black pawn, it should not be able
                        # to move 2 squares unless starting from its initial position on 7th rank
                        if piece == 'P' and offset == -20:
                            if square not in self.rank_2: break # if not on 7th rank, don't make this move as you can't move 2 squares from here
                            # remember white pawns go up the board in this config-
                            # so if we want to move the pawn 2 squares up we have to make sure there's nothing in the way
                            # i.e. we have a problem if there's something one square up- we can't use this offset to make the move
                            if self.board[square-10] != '.': break # don't make the move
                        # same for white pieces going down
                        if piece == 'p' and offset == 20:
                            if square not in self.rank_7: break # if not on 2nd rank, don't make this move- same as black above
                            if self.board[square+10] != '.': break
                        # if you are able to move your piece onto a square with a king, this is checkmate!
                        # shah mat! (the king's dead!) -> checkmate, no legal moves for the side moving as the game's over
                        if captured_piece in 'Kk': return []
                        # we add on to move_list the full information for this move
                        # where the piece came from
                        # where we moved it to
                        # what the piece itself is
                        # what was on the square we moved the piece to
                        move_list.append({
                            'source': square, 'target': target_square,
                            'piece': piece, 'captured': captured_piece
                        })
                        # otherwise we're still on the board so...
                        # set what is at the current square to which we have moved our piece to be the piece itself!
                        # TODO: self.board[target_square] = piece
                        # and of course we ensure that when we make this move using the offset
                        # we set the current square the piece was on to be empty
                        # TODO: self.board[square] = '.'
                        # this is for constructing a string that when printed out will result in a pretty printed board
                        # with pieces
                        # TODO: print(''.join([' ' + chess.pieces[p] for p in ''.join(chess.board)]), chess.side); input()
                        # reset the board
                        # TODO: self.board[target_square] = captured_piece
                        # TODO: self.board[square] = piece
                        # TODO: print(''.join([' ' + chess.pieces[p] for p in ''.join(chess.board)]), chess.side); input()
                        # if the square contains a piece of the opposite side, then we stop and move on to the next offset
                        # notice however that we first print and then move on to the next offset- this is
                        # because we did actually want to move our piece onto that square (and stop right there)
                        if self.colors[captured_piece] == self.side ^ 1: break # xor with 1 will flip the bit- so white=0^1=1 and black=1^1=0 i.e. RHS=opposite side
                        if piece in 'PpNnKk': break # this is because for pawns, kings and knights we must add the offset only once to list all moves
                        # i.e. after printing the new position and the reset position, we do not continue dding the offset, we just move on to the next offset
        # what will be returned is a list of all the legal moves
        return move_list

    # here we pass the move to make_move
    # the move is given in the form
    # {
    #     'source': square, 'target': target_square,
    #     'piece': piece, 'captured': captured_piece
    # }
    # as this all the information that defines the move
    def make_move(self, move):
        # we're moving the piece to the square at target_square as given in the move dictionary
        # so update that square on the board
        self.board[move['target']] = move['piece']
        # and of course replace the character at the original square 
        # by one representing an empty space
        self.board[move['source']] = '.'
        # but what if a pawn has a legal move that allows it to hit the 1st rank as black?
        # now we're going to use the default of queen promotion
        # i.e. stick a queen on the square to which the pawn moves to
        # (this essentially reads: if the piece is a black pawn and it's on 2nd rank, 
        # make wherever it's moving to a queen since pawns can only move up/down the board)
        if move['piece']=='P' and move['source'] in chess.rank_7:
            self.board[move['target']] == 'Q'
        # same for white
        if move['piece']=='p' and move['source'] in chess.rank_2:
            self.board[move['target']] == 'q'
        # pretty print what the board looks like now after making the move
        print("Move made: " + str(chess.pieces[move['piece']]) + " " + self.mapping_nums_to_squares(move['source']) + self.mapping_nums_to_squares(move['target']))
        print("Piece captured: " + str(chess.pieces[move['captured']]))
        print(''.join([chess.pieces[c] + " " if c not in [" ", "."] else ("  " if c == "." else c) for c in self.board]), chess.side); input()
        # we've made the move so it's the turn of the other side
        self.side ^= 1

    # again, the information given to take_back will be in the same form as for make_move
    def take_back(self, move):
        # move['captured'] stores the information of whatever
        # was actually at the square to which we moved our piece
        # i.e. what was at target_square before we set it to have the piece we moved onto it
        self.board[move['target']] = move['captured']
        # of course we also need to move the piece that was moved back to its original square, so...
        self.board[move['source']] = move['piece']
        # pretty print what the board looks like now
        print("Take back: " + str(chess.pieces[move['piece']] + " from " +\
               self.mapping_nums_to_squares(move['target']) + " to " + self.mapping_nums_to_squares(move['source'])))
        print(''.join([chess.pieces[c] + " " if c not in [" ", "."] else ("  " if c == "." else c) for c in self.board]), chess.side); input()
        # return the play to the side that it is to move
        # e.g. if it is black to play and white takebacks, it is now white's turn to play
        self.side ^= 1
    
    # returns, as a string, the file and rank of a given square on the board
    def mapping_nums_to_squares(self, num):
        # nums will be between 21-28, 31-38, etc. up to 91-98
        # we will calculate rank and file using ascii codes a:97, increasing to h:104
        #  97  98  99  100 101 102 103 104 105
        #  a   b   c   d   e   f   g   h   |
        #  81  82  83  84  85  86  87  88  |
        #  1   2   3   4   5   6   7   8 <──  (after mod 10)
        # nums ending 1 are file a, nums ending 8 are file h
        rank = 10-(num//10)
        file = chr(96+(num%10))
        return file+str(rank)

    # pretty print board
    def board_to_string(self):
        return (''.join([chess.pieces[c] + " " if c not in [" ", "."] else ("  " if c == "." else c) for c in self.board]))
chess = Chess('settings.json')

# here's what's going to happen here... generate_moves will give you all legal moves
# from the starting position specifically
# and for each of those moves, we're going to play them on the board and then take them back
# what you will see is that each legal move gets played, the board gets printed,
# and we put the piece back on the square it came from and print the board once more (yielding the start pos again)
# not that if we did this without taking back, then since chess.board does get modified, all the legal moves would be played
# without regard for what the board actually looks like as we're in the for loop (remember generate_moves provides legal moves from a given position)
# it is our duty to make sure that in a real game we update the legal moves as the position (chess.board) changes
# for move in chess.generate_moves():
#     chess.make_move(move)
#     chess.take_back(move)

# dictionary of the the directions in which each piece can move (just directions)
# for the knight, pawn and king we will list all possibilities which will determine, through addition,
# the new position of the piece on the board
# of course for the rook, king and queen there are too many possible additions
# so we simply list one step in a direction they can move in
# print(chess.directions)

def fen_to_vector(fen):
    """
    Convert a FEN string into a 64-dimensional vector using the following mapping:
      White: P=1, N=2, B=3, R=4, Q=5, K=6
      Black: p=-1, n=-2, b=-3, r=-4, q=-5, k=-6
      Empty: '.'=0

    The function reconstructs the board representation (with padding) in the same
    way as your Chess class, then extracts the 8x8 playable board.
    """
    piece_to_value = {
        'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
        'p': -1, 'n': -2, 'b': -3, 'r': -4, 'q': -5, 'k': -6,
        '.': 0
    }
    # The FEN's first field is the board layout.
    board_layout = fen.split()[0]
    # Create the padded board string exactly as in your __init__
    board_str = ((' ' * 9 + '\n') * 2 +
                 ' ' +
                 ''.join([c if not c.isdigit() else "." * int(c) 
                          for c in board_layout.replace('/', '\n ')]) + '\n' +
                 (' ' * 9 + '\n') * 2)
    board_list = list(board_str)
    vector = []
    # Playable rows are rows 2 through 9 (each row is 10 characters)
    for row in range(9, 1, -1):
        start = row * 10 + 1
        end = row * 10 + 9
        for idx in range(start, end):
            piece = board_list[idx]
            if piece not in "\n ":
                vector.append(piece_to_value.get(piece, 0))
    return np.array(vector)


def format_vector_as_matrix(vec):
    """Format a 64-element vector as an 8x8 matrix string."""
    mat = vec.reshape(8,8)
    lines = []
    for row in mat:
        lines.append(' '.join(f'{int(x):3d}' for x in row))
    return '\n'.join(lines)

def diff_vector_as_matrix(vec1, vec2):
    """Compute and format the element-wise difference between two vectors as an 8x8 matrix."""
    diff = vec1 - vec2
    return format_vector_as_matrix(diff)

def vector_to_latex(vec, caption=""):
    """
    Return a LaTeX tabular string representing the 8x8 matrix of the vector.
    """
    mat = vec.reshape(8,8)
    latex_str = "\\begin{tabular}{|" + "c|"*8 + "}\n\\hline\n"
    for row in mat:
        row_str = " & ".join(str(int(x)) for x in row) + " \\\\ \\hline\n"
        latex_str += row_str
    latex_str += "\\end{tabular}"
    if caption:
        latex_str = "\\caption{" + caption + "}\n" + latex_str
    return latex_str


def knn_predict(test_vector, training_data, k=3):
    distances = []
    # each point in the training data consists of its vector representation,
    # its evaluation and its fen string
    for train_vector, train_eval, train_fen in training_data:
        # for each data point we calculate how far it is from our test vector
        dist = np.linalg.norm(test_vector - train_vector)
        # and we add the point's information along with its distance from our test vector
        distances.append((dist, train_eval, train_fen, train_vector))  # include train_vector here
    # sort the positions according to how close they are according to our encoding
    distances.sort(key=lambda x: x[0])
    # take the top k positions
    top_neighbors = distances[:k]
    predicted_eval = sum([score for _, score, _, _ in top_neighbors]) / k if k > 0 else 0
    return predicted_eval, top_neighbors

def load_simplified_db(filename):
    """
    Load the simplified database from a JSONL file.
    
    Each line should be a JSON object with keys:
      - "fen": the FEN string
      - "eval": the extracted evaluation (in centipawns)
    
    Returns a list of tuples: (vector, eval, fen)
    """
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

def split_train_test(data, train_ratio=0.999):
    """
    Randomly shuffle and split the data into training and testing sets.
    """
    random.shuffle(data)
    split_index = int(len(data) * train_ratio)
    return data[:split_index], data[split_index:]

def main():
    filename = "simplified_db.jsonl"
    data = load_simplified_db(filename)
    print(f"Loaded {len(data)} positions from the simplified database.")
    
    train_data, test_data = split_train_test(data, train_ratio=0.8)
    print(f"Training set: {len(train_data)} positions, Test set: {len(test_data)} positions.")
    
    # Use only 0.1% of the test data for faster testing.
    test_sample_size = max(1, int(len(test_data) * 0.001))
    test_sample = random.sample(test_data, test_sample_size)
    
    # Prepare training set as list of (vector, eval, fen)
    training_set = [(vec, eval_val, fen) for vec, eval_val, fen in train_data]
    
    k = 3  # number of nearest neighbours to consider
    
    # Instantiate a Chess object for board visualization.
    chess = Chess('settings.json')
    
    for vec, actual_eval, fen in test_sample:
        predicted_eval, top_neighbors = knn_predict(vec, training_set, k=k)
        error = abs(predicted_eval - actual_eval)
        print("=" * 50)
        print(f"Original Position FEN: {fen}")
        print(f"Actual Eval: {actual_eval}, Predicted Eval (k-NN): {predicted_eval:.2f}, Error: {error:.2f}\n")
        
        # Show the board representation
        print("Original Board (ASCII):")
        chess.set_board_as_this_fen(fen)
        print(chess.board_to_string())
        orig_vector = fen_to_vector(fen)
        
        print("\nTop 3 Similar Positions:")
        for dist, neigh_eval, neigh_fen, neigh_vector in top_neighbors:
            print(f"\nDistance: {dist:.2f}, Eval: {neigh_eval}, FEN: {neigh_fen}")
            # Show neighbor board:
            chess.set_board_as_this_fen(neigh_fen)
            print(chess.board_to_string())
            # Show the difference vector (original - neighbor)
            diff = orig_vector - neigh_vector
            print("\nDifference (Original - Neighbor) Vector (8x8):")
            print(diff_vector_as_matrix(orig_vector, neigh_vector))
            print("-" * 30)
        print("=" * 50, "\n")
    
if __name__ == "__main__":
    main()