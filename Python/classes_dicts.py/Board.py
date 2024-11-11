class Board:
    """
    Class to represent a game board.

    Attributes:
        rows (int): number of rows
        cols (int): number of columns
        location_of_pieces (dictionary): maps player to list of locations of
            their pieces

    Methods:
        add_piece: add a piece represented by a string to the board
        on_board: determine if a location is within the bounds of the board
        adjacent: find coordinates of cells to the north, east, south, and west
            of the given location
    """
    def __init__(self, rows: int, cols: int):
        """
        Constructor for empty board

        Inputs:
            rows [int]: number of rows
            cols [int]: number of columns
        """

        self.rows = rows
        self.cols = cols
        self.location_of_pieces: dict = {}

    def add_piece(self, piece: str, location: tuple[int, int]) -> bool:
        """
        Add a piece represented by a string to the board.

        Inputs:
            piece [string]: the piece to add
            location [tuple]: the (row, column) location of where to add
                the piece
            
        Returns [bool]: True if the piece was added successfully,
            False otherwise
        """
        if not self.on_board(location):
            return False
        
        for player, locations in self.location_of_pieces.items():
            if location in locations:
                return False
        
        if piece in self.location_of_pieces:
            self.location_of_pieces[piece].append(location)
        else:
            self.location_of_pieces[piece] = [location]
        return True
        
    def on_board(self, location: tuple[int, int]) -> bool:
        """
        Determine if a location is within the bounds of the board

        Inputs:
            location [tuple]: the (row, column) location in question
            
        Returns [bool]: True if the location is on the board, False otherwise
        """
        row, col = location
        return row >= 0 and row < self.rows and \
               col >= 0 and col < self.cols

    def adjacent(self, location: tuple[int, int]) -> list[tuple[int, int]]:
        """
        Find coordinates of cells to the north, east, south, and west of the
            given location
            
        Inputs:
            location [tuple]: the (row, column) location in question
                
        Returns [list]: coordinates of adjacent spots
        """
        rv = []
        row, col = location
        
        deltas = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        for dr, dc in deltas:
            adj = (row + dr, col + dc)
            if self.on_board(adj):
                rv.append(adj)
        
        return rv

