from fastapi import FastAPI, APIRouter, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union, Tuple
import uuid
from datetime import datetime
from enum import Enum
import json
import copy

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Enums
class PieceType(str, Enum):
    PAWN = "pawn"
    ROOK = "rook"
    KNIGHT = "knight"
    BISHOP = "bishop"
    QUEEN = "queen"
    KING = "king"

class PieceColor(str, Enum):
    WHITE = "white"
    BLACK = "black"

class MoveType(str, Enum):
    CLASSICAL = "classical"
    QUANTUM = "quantum"
    MEASURE = "measure"

class GameStatus(str, Enum):
    ACTIVE = "active"
    CHECK_WHITE = "check_white"
    CHECK_BLACK = "check_black"
    CHECKMATE_WHITE = "checkmate_white"
    CHECKMATE_BLACK = "checkmate_black"
    STALEMATE = "stalemate"

# Models
class QuantumPiece(BaseModel):
    piece_type: PieceType
    color: PieceColor
    probability: float = 0.5
    quantum_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

class Square(BaseModel):
    classical_piece: Optional[QuantumPiece] = None
    quantum_pieces: List[QuantumPiece] = Field(default_factory=list)
    
    def is_empty(self) -> bool:
        return self.classical_piece is None and len(self.quantum_pieces) == 0
    
    def has_classical_piece(self) -> bool:
        return self.classical_piece is not None
    
    def has_quantum_pieces(self) -> bool:
        return len(self.quantum_pieces) > 0

class Move(BaseModel):
    from_pos: Tuple[int, int]
    to_pos: Tuple[int, int]
    move_type: MoveType
    piece_type: PieceType
    color: PieceColor
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    quantum_id: Optional[str] = None

class Game(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    board: List[List[Square]] = Field(default_factory=lambda: [[Square() for _ in range(8)] for _ in range(8)])
    current_player: PieceColor = PieceColor.WHITE
    move_history: List[Move] = Field(default_factory=list)
    status: GameStatus = GameStatus.ACTIVE
    white_superpositions: int = 0
    black_superpositions: int = 0
    is_vs_ai: bool = False
    ai_color: Optional[PieceColor] = None
    ai_difficulty: Optional[str] = None  # "easy" or "medium"
    created_at: datetime = Field(default_factory=datetime.utcnow)

class MoveRequest(BaseModel):
    from_pos: Tuple[int, int]
    to_pos: Tuple[int, int]
    move_type: MoveType
    quantum_id: Optional[str] = None

class MeasureRequest(BaseModel):
    positions: List[Tuple[int, int]]

class GameCreateRequest(BaseModel):
    is_vs_ai: bool = False
    ai_color: Optional[PieceColor] = None
    ai_difficulty: Optional[str] = None

# Chess logic utilities
def create_initial_board() -> List[List[Square]]:
    """Create the initial chess board setup"""
    board = [[Square() for _ in range(8)] for _ in range(8)]
    
    # Set up pawns
    for col in range(8):
        board[1][col].classical_piece = QuantumPiece(piece_type=PieceType.PAWN, color=PieceColor.BLACK)
        board[6][col].classical_piece = QuantumPiece(piece_type=PieceType.PAWN, color=PieceColor.WHITE)
    
    # Set up back ranks
    piece_order = [PieceType.ROOK, PieceType.KNIGHT, PieceType.BISHOP, PieceType.QUEEN, 
                   PieceType.KING, PieceType.BISHOP, PieceType.KNIGHT, PieceType.ROOK]
    
    for col in range(8):
        board[0][col].classical_piece = QuantumPiece(piece_type=piece_order[col], color=PieceColor.BLACK)
        board[7][col].classical_piece = QuantumPiece(piece_type=piece_order[col], color=PieceColor.WHITE)
    
    return board

def is_valid_position(row: int, col: int) -> bool:
    """Check if position is within board bounds"""
    return 0 <= row < 8 and 0 <= col < 8

def get_piece_at_position(board: List[List[Square]], row: int, col: int) -> Optional[QuantumPiece]:
    """Get the classical piece at a position, if any"""
    if not is_valid_position(row, col):
        return None
    return board[row][col].classical_piece

def is_path_clear(board: List[List[Square]], from_row: int, from_col: int, to_row: int, to_col: int) -> bool:
    """Check if path is clear for sliding pieces (rook, bishop, queen)"""
    row_dir = 0 if from_row == to_row else (1 if to_row > from_row else -1)
    col_dir = 0 if from_col == to_col else (1 if to_col > from_col else -1)
    
    current_row, current_col = from_row + row_dir, from_col + col_dir
    
    while (current_row, current_col) != (to_row, to_col):
        if not board[current_row][current_col].is_empty():
            return False
        current_row += row_dir
        current_col += col_dir
    
    return True

def is_valid_classical_move(board: List[List[Square]], from_row: int, from_col: int, to_row: int, to_col: int, piece: QuantumPiece) -> bool:
    """Validate classical chess move"""
    if not is_valid_position(to_row, to_col):
        return False
    
    # Can't capture own piece
    target_piece = get_piece_at_position(board, to_row, to_col)
    if target_piece and target_piece.color == piece.color:
        return False
    
    row_diff = abs(to_row - from_row)
    col_diff = abs(to_col - from_col)
    
    # Piece-specific movement validation
    if piece.piece_type == PieceType.PAWN:
        direction = -1 if piece.color == PieceColor.WHITE else 1
        start_row = 6 if piece.color == PieceColor.WHITE else 1
        
        # Forward move
        if from_col == to_col:
            if to_row == from_row + direction and not target_piece:
                return True
            if from_row == start_row and to_row == from_row + 2 * direction and not target_piece:
                return True
        # Diagonal capture
        elif col_diff == 1 and to_row == from_row + direction and target_piece:
            return True
        return False
    
    elif piece.piece_type == PieceType.ROOK:
        if (row_diff == 0 or col_diff == 0) and is_path_clear(board, from_row, from_col, to_row, to_col):
            return True
    
    elif piece.piece_type == PieceType.BISHOP:
        if row_diff == col_diff and is_path_clear(board, from_row, from_col, to_row, to_col):
            return True
    
    elif piece.piece_type == PieceType.QUEEN:
        if ((row_diff == 0 or col_diff == 0) or (row_diff == col_diff)) and is_path_clear(board, from_row, from_col, to_row, to_col):
            return True
    
    elif piece.piece_type == PieceType.KNIGHT:
        if (row_diff == 2 and col_diff == 1) or (row_diff == 1 and col_diff == 2):
            return True
    
    elif piece.piece_type == PieceType.KING:
        if row_diff <= 1 and col_diff <= 1:
            return True
    
    return False

def find_king(board: List[List[Square]], color: PieceColor) -> Optional[Tuple[int, int]]:
    """Find the king's position for a given color"""
    for row in range(8):
        for col in range(8):
            piece = board[row][col].classical_piece
            if piece and piece.piece_type == PieceType.KING and piece.color == color:
                return (row, col)
    return None

def is_square_attacked(board: List[List[Square]], row: int, col: int, by_color: PieceColor) -> bool:
    """Check if a square is attacked by pieces of given color"""
    for r in range(8):
        for c in range(8):
            piece = board[r][c].classical_piece
            if piece and piece.color == by_color:
                if is_valid_classical_move(board, r, c, row, col, piece):
                    return True
    return False

def is_in_check(board: List[List[Square]], color: PieceColor) -> bool:
    """Check if the king of given color is in check"""
    king_pos = find_king(board, color)
    if not king_pos:
        return False
    
    opponent_color = PieceColor.BLACK if color == PieceColor.WHITE else PieceColor.WHITE
    return is_square_attacked(board, king_pos[0], king_pos[1], opponent_color)

def get_legal_moves(board: List[List[Square]], color: PieceColor) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Get all legal moves for a color"""
    legal_moves = []
    
    for from_row in range(8):
        for from_col in range(8):
            piece = board[from_row][from_col].classical_piece
            if piece and piece.color == color:
                for to_row in range(8):
                    for to_col in range(8):
                        if is_valid_classical_move(board, from_row, from_col, to_row, to_col, piece):
                            # Test if this move leaves king in check
                            test_board = copy.deepcopy(board)
                            test_board[to_row][to_col].classical_piece = piece
                            test_board[from_row][from_col].classical_piece = None
                            
                            if not is_in_check(test_board, color):
                                legal_moves.append(((from_row, from_col), (to_row, to_col)))
    
    return legal_moves

def evaluate_board_position(board: List[List[Square]], color: PieceColor) -> int:
    """Simple board evaluation for AI"""
    score = 0
    piece_values = {
        PieceType.PAWN: 1,
        PieceType.KNIGHT: 3,
        PieceType.BISHOP: 3,
        PieceType.ROOK: 5,
        PieceType.QUEEN: 9,
        PieceType.KING: 1000
    }
    
    for row in range(8):
        for col in range(8):
            piece = board[row][col].classical_piece
            if piece:
                value = piece_values.get(piece.piece_type, 0)
                if piece.color == color:
                    score += value
                else:
                    score -= value
    
    return score

def score_move(game: Game, from_pos: Tuple[int, int], to_pos: Tuple[int, int], move_type: str) -> float:
    """Score a potential move for AI"""
    from_row, from_col = from_pos
    to_row, to_col = to_pos
    
    # Create test game state
    test_game = copy.deepcopy(game)
    piece = test_game.board[from_row][from_col].classical_piece
    if not piece:
        return -1000
    
    score = 0
    
    # Basic move scoring
    target_piece = test_game.board[to_row][to_col].classical_piece
    if target_piece and target_piece.color != piece.color:
        score += 3  # Capture bonus
    
    # Pawn advancement bonus
    if piece.piece_type == PieceType.PAWN:
        direction = -1 if piece.color == PieceColor.WHITE else 1
        if (to_row - from_row) * direction > 0:
            score += 1
    
    # Simulate the move
    if move_type == "classical":
        test_game.board[from_row][from_col].classical_piece = None
        test_game.board[to_row][to_col].classical_piece = piece
        
        # Check if this leaves king in check (heavily penalize)
        if is_in_check(test_game.board, piece.color):
            score -= 100
            
    elif move_type == "quantum":
        # For quantum moves, simulate multiple collapses
        collapse_scores = []
        
        for _ in range(5):  # Simulate 5 random collapses
            sim_game = copy.deepcopy(test_game)
            
            # Create quantum superposition
            quantum_id = str(uuid.uuid4())
            quantum_piece_1 = QuantumPiece(
                piece_type=piece.piece_type,
                color=piece.color,
                probability=0.5,
                quantum_id=quantum_id
            )
            quantum_piece_2 = QuantumPiece(
                piece_type=piece.piece_type,
                color=piece.color,
                probability=0.5,
                quantum_id=quantum_id
            )
            
            sim_game.board[from_row][from_col].classical_piece = None
            sim_game.board[from_row][from_col].quantum_pieces.append(quantum_piece_1)
            sim_game.board[to_row][to_col].quantum_pieces.append(quantum_piece_2)
            
            # Randomly collapse to one position
            import random
            if random.random() < 0.5:
                # Collapse to original position
                sim_game.board[from_row][from_col].quantum_pieces = []
                sim_game.board[from_row][from_col].classical_piece = piece
                sim_game.board[to_row][to_col].quantum_pieces = []
            else:
                # Collapse to new position
                sim_game.board[from_row][from_col].quantum_pieces = []
                sim_game.board[to_row][to_col].quantum_pieces = []
                sim_game.board[to_row][to_col].classical_piece = piece
            
            # Evaluate this collapsed state
            board_score = evaluate_board_position(sim_game.board, piece.color)
            collapse_scores.append(board_score)
        
        # Average the collapse scores
        score += sum(collapse_scores) / len(collapse_scores) / 100  # Scale down
    
    return score

def get_ai_move(game: Game) -> Optional[Dict]:
    """Generate AI move based on difficulty"""
    if not game.is_vs_ai or game.current_player != game.ai_color:
        return None
    
    # Get all legal moves
    legal_moves = get_legal_moves(game.board, game.current_player)
    if not legal_moves:
        return None
    
    # Check if quantum moves are available
    max_superpositions = 1
    current_superpositions = game.white_superpositions if game.current_player == PieceColor.WHITE else game.black_superpositions
    can_make_quantum = current_superpositions < max_superpositions
    
    # Generate all possible moves (classical and quantum)
    possible_moves = []
    
    # Add classical moves
    for from_pos, to_pos in legal_moves:
        possible_moves.append({
            "from_pos": from_pos,
            "to_pos": to_pos,
            "move_type": "classical"
        })
    
    # Add quantum moves if available
    if can_make_quantum:
        for from_pos, to_pos in legal_moves:
            to_row, to_col = to_pos
            # Only quantum move to empty squares
            if game.board[to_row][to_col].is_empty():
                possible_moves.append({
                    "from_pos": from_pos,
                    "to_pos": to_pos,
                    "move_type": "quantum"
                })
    
    if not possible_moves:
        return None
    
    # Select move based on difficulty
    if game.ai_difficulty == "easy":
        # Random move
        import random
        return random.choice(possible_moves)
    
    elif game.ai_difficulty == "medium":
        # Score-based move selection
        best_move = None
        best_score = float('-inf')
        
        for move in possible_moves:
            score = score_move(game, move["from_pos"], move["to_pos"], move["move_type"])
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move
    
    # Fallback to random
    import random
    return random.choice(possible_moves)

def measure_quantum_pieces(game: Game, positions: List[Tuple[int, int]]) -> Game:
    """Measure and collapse quantum pieces at given positions"""
    import random
    
    for row, col in positions:
        if not is_valid_position(row, col):
            continue
        
        square = game.board[row][col]
        if square.has_quantum_pieces():
            # For now, randomly collapse to one of the quantum pieces
            chosen_piece = random.choice(square.quantum_pieces)
            
            # Remove all quantum pieces and set as classical
            square.quantum_pieces = []
            square.classical_piece = chosen_piece
            
            # Find and remove the other superposition
            for r in range(8):
                for c in range(8):
                    if (r, c) != (row, col):
                        other_square = game.board[r][c]
                        other_square.quantum_pieces = [
                            p for p in other_square.quantum_pieces 
                            if p.quantum_id != chosen_piece.quantum_id
                        ]
            
            # Decrease superposition count
            if chosen_piece.color == PieceColor.WHITE:
                game.white_superpositions = max(0, game.white_superpositions - 1)
            else:
                game.black_superpositions = max(0, game.black_superpositions - 1)
    
    return game

# API Routes
@api_router.post("/game", response_model=Game)
async def create_game(request: Optional[GameCreateRequest] = None):
    """Create a new quantum chess game"""
    game = Game()
    game.board = create_initial_board()
    
    if request:
        game.is_vs_ai = request.is_vs_ai
        game.ai_color = request.ai_color
        game.ai_difficulty = request.ai_difficulty
    
    await db.games.insert_one(game.dict())
    return game

@api_router.get("/game/{game_id}", response_model=Game)
async def get_game(game_id: str):
    """Get game state"""
    game_doc = await db.games.find_one({"id": game_id})
    if not game_doc:
        raise HTTPException(status_code=404, detail="Game not found")
    
    return Game(**game_doc)

@api_router.post("/game/{game_id}/move")
async def make_move(game_id: str, move_request: MoveRequest):
    """Make a move in the game"""
    game_doc = await db.games.find_one({"id": game_id})
    if not game_doc:
        raise HTTPException(status_code=404, detail="Game not found")
    
    game = Game(**game_doc)
    from_row, from_col = move_request.from_pos
    to_row, to_col = move_request.to_pos
    
    if not is_valid_position(from_row, from_col) or not is_valid_position(to_row, to_col):
        raise HTTPException(status_code=400, detail="Invalid position")
    
    from_square = game.board[from_row][from_col]
    to_square = game.board[to_row][to_col]
    
    # Get the piece to move
    piece = None
    if move_request.move_type == MoveType.CLASSICAL:
        piece = from_square.classical_piece
        if not piece or piece.color != game.current_player:
            raise HTTPException(status_code=400, detail="No valid piece to move")
        
        # Validate move
        if not is_valid_classical_move(game.board, from_row, from_col, to_row, to_col, piece):
            raise HTTPException(status_code=400, detail="Invalid move")
        
        # Make the move
        from_square.classical_piece = None
        to_square.classical_piece = piece
        
    elif move_request.move_type == MoveType.QUANTUM:
        piece = from_square.classical_piece
        if not piece or piece.color != game.current_player:
            raise HTTPException(status_code=400, detail="No valid piece to move")
        
        # Check superposition limit (beginner mode: 1 per player)
        max_superpositions = 1
        current_superpositions = game.white_superpositions if game.current_player == PieceColor.WHITE else game.black_superpositions
        
        if current_superpositions >= max_superpositions:
            raise HTTPException(status_code=400, detail="Maximum superpositions reached")
        
        # Validate quantum move
        if not is_valid_classical_move(game.board, from_row, from_col, to_row, to_col, piece):
            raise HTTPException(status_code=400, detail="Invalid quantum move")
        
        if not to_square.is_empty():
            raise HTTPException(status_code=400, detail="Cannot create quantum superposition on occupied square")
        
        # Create quantum superposition
        quantum_id = str(uuid.uuid4())
        quantum_piece_1 = QuantumPiece(
            piece_type=piece.piece_type,
            color=piece.color,
            probability=0.5,
            quantum_id=quantum_id
        )
        quantum_piece_2 = QuantumPiece(
            piece_type=piece.piece_type,
            color=piece.color,
            probability=0.5,
            quantum_id=quantum_id
        )
        
        # Remove classical piece and add quantum pieces
        from_square.classical_piece = None
        from_square.quantum_pieces.append(quantum_piece_1)
        to_square.quantum_pieces.append(quantum_piece_2)
        
        # Update superposition count
        if game.current_player == PieceColor.WHITE:
            game.white_superpositions += 1
        else:
            game.black_superpositions += 1
    
    # Record the move
    move = Move(
        from_pos=move_request.from_pos,
        to_pos=move_request.to_pos,
        move_type=move_request.move_type,
        piece_type=piece.piece_type,
        color=piece.color,
        quantum_id=getattr(piece, 'quantum_id', None) if hasattr(piece, 'quantum_id') else None
    )
    game.move_history.append(move)
    
    # Check for game status updates
    opponent_color = PieceColor.BLACK if game.current_player == PieceColor.WHITE else PieceColor.WHITE
    
    if is_in_check(game.board, opponent_color):
        game.status = GameStatus.CHECK_BLACK if opponent_color == PieceColor.BLACK else GameStatus.CHECK_WHITE
        
        # Check for checkmate
        legal_moves = get_legal_moves(game.board, opponent_color)
        if not legal_moves:
            game.status = GameStatus.CHECKMATE_BLACK if opponent_color == PieceColor.BLACK else GameStatus.CHECKMATE_WHITE
    else:
        game.status = GameStatus.ACTIVE
    
    # Switch turns
    game.current_player = opponent_color
    
    # Update database
    await db.games.replace_one({"id": game_id}, game.dict())
    
    return {"success": True, "game": game}

@api_router.post("/game/{game_id}/measure")
async def measure_pieces(game_id: str, measure_request: MeasureRequest):
    """Measure quantum pieces at specified positions"""
    game_doc = await db.games.find_one({"id": game_id})
    if not game_doc:
        raise HTTPException(status_code=404, detail="Game not found")
    
    game = Game(**game_doc)
    game = measure_quantum_pieces(game, measure_request.positions)
    
    # Update database
    await db.games.replace_one({"id": game_id}, game.dict())
    
    return {"success": True, "game": game}

@api_router.get("/game/{game_id}/legal-moves")
async def get_game_legal_moves(game_id: str):
    """Get legal moves for current player"""
    game_doc = await db.games.find_one({"id": game_id})
    if not game_doc:
        raise HTTPException(status_code=404, detail="Game not found")
    
    game = Game(**game_doc)
    legal_moves = get_legal_moves(game.board, game.current_player)
    
    return {"legal_moves": legal_moves, "current_player": game.current_player}

@api_router.post("/game/{game_id}/ai_move")
async def make_ai_move(game_id: str):
    """Generate and execute an AI move"""
    game_doc = await db.games.find_one({"id": game_id})
    if not game_doc:
        raise HTTPException(status_code=404, detail="Game not found")
    
    game = Game(**game_doc)
    
    if not game.is_vs_ai:
        raise HTTPException(status_code=400, detail="Game is not vs AI")
    
    if game.current_player != game.ai_color:
        raise HTTPException(status_code=400, detail="Not AI's turn")
    
    # Check if game is already over
    if game.status in [GameStatus.CHECKMATE_WHITE, GameStatus.CHECKMATE_BLACK, GameStatus.STALEMATE]:
        return {"success": False, "message": "Game is already over", "game": game}
    
    # Get AI move
    ai_move = get_ai_move(game)
    if not ai_move:
        return {"success": False, "message": "No legal moves available", "game": game}
    
    # Execute the AI move using existing logic
    from_row, from_col = ai_move["from_pos"]
    to_row, to_col = ai_move["to_pos"]
    move_type = ai_move["move_type"]
    
    from_square = game.board[from_row][from_col]
    to_square = game.board[to_row][to_col]
    piece = from_square.classical_piece
    
    if not piece:
        raise HTTPException(status_code=500, detail="AI selected invalid move")
    
    # Execute the move
    if move_type == "classical":
        from_square.classical_piece = None
        to_square.classical_piece = piece
        
    elif move_type == "quantum":
        # Create quantum superposition
        quantum_id = str(uuid.uuid4())
        quantum_piece_1 = QuantumPiece(
            piece_type=piece.piece_type,
            color=piece.color,
            probability=0.5,
            quantum_id=quantum_id
        )
        quantum_piece_2 = QuantumPiece(
            piece_type=piece.piece_type,
            color=piece.color,
            probability=0.5,
            quantum_id=quantum_id
        )
        
        from_square.classical_piece = None
        from_square.quantum_pieces.append(quantum_piece_1)
        to_square.quantum_pieces.append(quantum_piece_2)
        
        # Update superposition count
        if game.current_player == PieceColor.WHITE:
            game.white_superpositions += 1
        else:
            game.black_superpositions += 1
    
    # Record the move
    move = Move(
        from_pos=ai_move["from_pos"],
        to_pos=ai_move["to_pos"],
        move_type=MoveType(move_type),
        piece_type=piece.piece_type,
        color=piece.color
    )
    game.move_history.append(move)
    
    # Check for game status updates
    opponent_color = PieceColor.BLACK if game.current_player == PieceColor.WHITE else PieceColor.WHITE
    
    if is_in_check(game.board, opponent_color):
        game.status = GameStatus.CHECK_BLACK if opponent_color == PieceColor.BLACK else GameStatus.CHECK_WHITE
        
        # Check for checkmate
        legal_moves = get_legal_moves(game.board, opponent_color)
        if not legal_moves:
            game.status = GameStatus.CHECKMATE_BLACK if opponent_color == PieceColor.BLACK else GameStatus.CHECKMATE_WHITE
    else:
        game.status = GameStatus.ACTIVE
    
    # Switch turns
    game.current_player = opponent_color
    
    # Update database
    await db.games.replace_one({"id": game_id}, game.dict())
    
    return {
        "success": True, 
        "game": game, 
        "ai_move": {
            "from_pos": ai_move["from_pos"],
            "to_pos": ai_move["to_pos"],
            "move_type": move_type,
            "piece_type": piece.piece_type.value,
            "color": piece.color.value
        }
    }

# Health check
@api_router.get("/")
async def root():
    return {"message": "Quantum Chess API Ready"}

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()