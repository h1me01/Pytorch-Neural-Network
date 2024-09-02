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
    stm = 0 if 'w' in fen else 1

    input1 = np.zeros(768) 
    input2 = np.zeros(768) 
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
                idx1 = index(sqr, pc, pt, 0) # white view
                idx2 = index(sqr, pc, pt, 1) # black view
                input1[idx1] = 1
                input2[idx2] = 1
                file += 1 

    if stm == 1:
        input1, input2 = input2, input1

    return input1, input2