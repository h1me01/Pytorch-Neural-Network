import numpy as np

piece_idx = {
    'p': 0,  
    'n': 1,  
    'b': 2,  
    'r': 3,  
    'q': 4,  
    'k': 5   
}

def index(sqr, pc, pt, view):
    if view != 0:
        sqr = sqr ^ 56
    return sqr + pt * 64 + (pc != view) * 64 * 6

def get(fen):
    view = 0 if 'w' in fen else 1

    sparse_input = np.zeros(768) 
    fen_board = fen.split(' ')[0] 
    fen_rows = fen_board.split('/')

    for rank in range(8):
        row = fen_rows[7 - rank] 
        file = 0  

        for char in row:
            if char.isdigit():
                file += int(char) 
            else:
                p = char.lower()
                pc = 0 if char.isupper() else 1
                pt = piece_idx[p]
                sqr = rank * 8 + file
                idx = index(sqr, pc, pt, view)
                sparse_input[idx] = 1
                file += 1 

    return sparse_input