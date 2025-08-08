import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';
import { Card } from './components/ui/card';
import { Button } from './components/ui/button';
import { Badge } from './components/ui/badge';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

// Chess piece Unicode symbols
const PIECE_SYMBOLS = {
  white: {
    king: '♔',
    queen: '♕',
    rook: '♖',
    bishop: '♗',
    knight: '♘',
    pawn: '♙'
  },
  black: {
    king: '♚',
    queen: '♛',
    rook: '♜',
    bishop: '♝',
    knight: '♞',
    pawn: '♟'
  }
};

const ChessSquare = ({ 
  square, 
  position, 
  isLight, 
  isSelected, 
  isHighlighted, 
  isLegalMove,
  onClick 
}) => {
  const [row, col] = position;
  
  const getSquareClasses = () => {
    let classes = 'chess-square w-16 h-16 flex items-center justify-center text-4xl font-bold cursor-pointer relative transition-all duration-200 ';
    
    if (isLight) {
      classes += 'bg-amber-100 ';
    } else {
      classes += 'bg-amber-800 ';
    }
    
    if (isSelected) {
      classes += 'ring-4 ring-blue-500 ';
    }
    
    if (isHighlighted) {
      classes += 'bg-yellow-300 ';
    }
    
    if (isLegalMove) {
      classes += 'ring-2 ring-green-500 ';
    }
    
    return classes;
  };

  const renderPiece = () => {
    if (square.classical_piece) {
      const piece = square.classical_piece;
      return (
        <span className="piece classical-piece">
          {PIECE_SYMBOLS[piece.color][piece.piece_type]}
        </span>
      );
    }
    
    if (square.quantum_pieces && square.quantum_pieces.length > 0) {
      return square.quantum_pieces.map((piece, index) => (
        <span 
          key={piece.quantum_id} 
          className={`piece quantum-piece absolute opacity-60 ${index === 0 ? 'top-1 left-1' : 'bottom-1 right-1'}`}
          style={{
            textShadow: '0 0 10px rgba(147, 51, 234, 0.8)',
            color: index === 0 ? '#8b5cf6' : '#a78bfa'
          }}
        >
          {PIECE_SYMBOLS[piece.color][piece.piece_type]}
        </span>
      ));
    }
    
    return null;
  };

  return (
    <div 
      className={getSquareClasses()}
      onClick={() => onClick(row, col)}
    >
      {renderPiece()}
      {isLegalMove && (
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="w-4 h-4 bg-green-500 rounded-full opacity-70"></div>
        </div>
      )}
    </div>
  );
};

const ChessBoard = ({ game, onSquareClick, selectedSquare, legalMoves }) => {
  const isLegalMove = (row, col) => {
    if (!selectedSquare || !legalMoves) return false;
    return legalMoves.some(([from, to]) => 
      from[0] === selectedSquare[0] && from[1] === selectedSquare[1] &&
      to[0] === row && to[1] === col
    );
  };

  return (
    <div className="chess-board grid grid-cols-8 gap-0 border-4 border-amber-900 shadow-2xl bg-amber-900 p-2 rounded-lg">
      {game.board.map((row, rowIndex) => 
        row.map((square, colIndex) => (
          <ChessSquare
            key={`${rowIndex}-${colIndex}`}
            square={square}
            position={[rowIndex, colIndex]}
            isLight={(rowIndex + colIndex) % 2 === 0}
            isSelected={selectedSquare && selectedSquare[0] === rowIndex && selectedSquare[1] === colIndex}
            isLegalMove={isLegalMove(rowIndex, colIndex)}
            onClick={onSquareClick}
          />
        ))
      )}
    </div>
  );
};

const GameModeModal = ({ isOpen, onClose, onCreateGame }) => {
  const [gameMode, setGameMode] = useState('human');
  const [playerColor, setPlayerColor] = useState('white');
  const [aiDifficulty, setAiDifficulty] = useState('easy');

  const handleCreateGame = () => {
    const gameConfig = {
      is_vs_ai: gameMode === 'ai',
      ai_color: gameMode === 'ai' ? (playerColor === 'white' ? 'black' : 'white') : null,
      ai_difficulty: gameMode === 'ai' ? aiDifficulty : null
    };
    onCreateGame(gameConfig);
    onClose();
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <Card className="p-8 max-w-md w-full mx-4">
        <h2 className="text-2xl font-bold text-center mb-6">New Quantum Chess Game</h2>
        
        <div className="space-y-6">
          {/* Game Mode Selection */}
          <div>
            <h3 className="text-lg font-semibold mb-3">Game Mode</h3>
            <div className="space-y-2">
              <div className="flex items-center space-x-2">
                <input
                  type="radio"
                  id="human"
                  name="gameMode"
                  value="human"
                  checked={gameMode === 'human'}
                  onChange={(e) => setGameMode(e.target.value)}
                  className="w-4 h-4"
                />
                <label htmlFor="human" className="text-sm font-medium">Human vs Human</label>
              </div>
              <div className="flex items-center space-x-2">
                <input
                  type="radio"
                  id="ai"
                  name="gameMode"
                  value="ai"
                  checked={gameMode === 'ai'}
                  onChange={(e) => setGameMode(e.target.value)}
                  className="w-4 h-4"
                />
                <label htmlFor="ai" className="text-sm font-medium">Human vs Computer</label>
              </div>
            </div>
          </div>

          {/* AI Settings */}
          {gameMode === 'ai' && (
            <>
              <div>
                <h3 className="text-lg font-semibold mb-3">Play As</h3>
                <div className="space-y-2">
                  <div className="flex items-center space-x-2">
                    <input
                      type="radio"
                      id="white"
                      name="playerColor"
                      value="white"
                      checked={playerColor === 'white'}
                      onChange={(e) => setPlayerColor(e.target.value)}
                      className="w-4 h-4"
                    />
                    <label htmlFor="white" className="text-sm font-medium">White (First Move)</label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <input
                      type="radio"
                      id="black"
                      name="playerColor"
                      value="black"
                      checked={playerColor === 'black'}
                      onChange={(e) => setPlayerColor(e.target.value)}
                      className="w-4 h-4"
                    />
                    <label htmlFor="black" className="text-sm font-medium">Black (Second Move)</label>
                  </div>
                </div>
              </div>

              <div>
                <h3 className="text-lg font-semibold mb-3">AI Difficulty</h3>
                <div className="space-y-2">
                  <div className="flex items-center space-x-2">
                    <input
                      type="radio"
                      id="easy"
                      name="aiDifficulty"
                      value="easy"
                      checked={aiDifficulty === 'easy'}
                      onChange={(e) => setAiDifficulty(e.target.value)}
                      className="w-4 h-4"
                    />
                    <label htmlFor="easy" className="text-sm font-medium">Easy (Random Moves)</label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <input
                      type="radio"
                      id="medium"
                      name="aiDifficulty"
                      value="medium"
                      checked={aiDifficulty === 'medium'}
                      onChange={(e) => setAiDifficulty(e.target.value)}
                      className="w-4 h-4"
                    />
                    <label htmlFor="medium" className="text-sm font-medium">Medium (Strategic)</label>
                  </div>
                </div>
              </div>
            </>
          )}

          <div className="flex space-x-3 pt-4">
            <Button onClick={onClose} variant="outline" className="flex-1">
              Cancel
            </Button>
            <Button onClick={handleCreateGame} className="flex-1">
              Create Game
            </Button>
          </div>
        </div>
      </Card>
    </div>
  );
};

const GameControls = ({ 
  game, 
  selectedSquare, 
  onClassicalMove, 
  onQuantumMove, 
  onMeasure, 
  onNewGame,
  aiThinking 
}) => {
  const canMakeQuantumMove = () => {
    if (!selectedSquare) return false;
    const maxSuperpositions = 1; // Beginner mode
    const currentSuperpositions = game.current_player === 'white' 
      ? game.white_superpositions 
      : game.black_superpositions;
    return currentSuperpositions < maxSuperpositions;
  };

  const hasQuantumPieces = () => {
    return game.board.some(row => 
      row.some(square => square.quantum_pieces && square.quantum_pieces.length > 0)
    );
  };

  return (
    <Card className="p-6 space-y-4 bg-gradient-to-br from-indigo-50 to-purple-50">
      <div className="text-center">
        <h2 className="text-2xl font-bold text-gray-800 mb-2">Quantum Chess</h2>
        <div className="flex items-center justify-center space-x-2 mb-2">
          <Badge variant={game.current_player === 'white' ? 'default' : 'secondary'}>
            Current Player: {game.current_player.charAt(0).toUpperCase() + game.current_player.slice(1)}
            {game.is_vs_ai && game.current_player === game.ai_color && ' (AI)'}
          </Badge>
          <Badge variant={game.status === 'active' ? 'default' : 'destructive'}>
            Status: {game.status.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
          </Badge>
        </div>
        {game.is_vs_ai && (
          <div className="text-sm text-gray-600">
            Playing vs AI ({game.ai_difficulty}) • You are {game.ai_color === 'white' ? 'Black' : 'White'}
          </div>
        )}
        {aiThinking && (
          <div className="flex items-center justify-center space-x-2 text-blue-600 animate-pulse">
            <div className="w-2 h-2 bg-blue-600 rounded-full animate-bounce"></div>
            <span className="text-sm font-medium">AI is thinking...</span>
            <div className="w-2 h-2 bg-blue-600 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
          </div>
        )}
      </div>

      <div className="grid grid-cols-2 gap-2 text-sm">
        <div className="text-center">
          <div className="font-medium">White Superpositions</div>
          <div className="text-lg font-bold text-blue-600">{game.white_superpositions}/1</div>
        </div>
        <div className="text-center">
          <div className="font-medium">Black Superpositions</div>
          <div className="text-lg font-bold text-red-600">{game.black_superpositions}/1</div>
        </div>
      </div>

      <div className="space-y-2">
        <Button 
          onClick={onClassicalMove} 
          disabled={!selectedSquare || aiThinking || (game.is_vs_ai && game.current_player === game.ai_color)}
          className="w-full"
          variant="default"
        >
          Classical Move
        </Button>
        
        <Button 
          onClick={onQuantumMove} 
          disabled={!selectedSquare || !canMakeQuantumMove() || aiThinking || (game.is_vs_ai && game.current_player === game.ai_color)}
          className="w-full"
          variant="outline"
        >
          Quantum Move {!canMakeQuantumMove() ? '(Max Reached)' : ''}
        </Button>
        
        <Button 
          onClick={onMeasure} 
          disabled={!hasQuantumPieces() || aiThinking || (game.is_vs_ai && game.current_player === game.ai_color)}
          className="w-full"
          variant="secondary"
        >
          Measure Quantum Pieces
        </Button>
        
        <Button 
          onClick={onNewGame} 
          className="w-full"
          variant="destructive"
        >
          New Game
        </Button>
      </div>

      <div className="text-xs text-gray-600 space-y-1">
        <div><strong>Beginner Mode:</strong> Max 1 superposition per player</div>
        <div><strong>Quantum Move:</strong> Split piece into 50/50 probability</div>
        <div><strong>Measure:</strong> Collapse quantum states to classical</div>
      </div>
    </Card>
  );
};

const MoveHistory = ({ moves }) => {
  return (
    <Card className="p-4 max-h-64 overflow-y-auto">
      <h3 className="text-lg font-bold mb-2">Move History</h3>
      <div className="space-y-1 text-sm">
        {moves.length === 0 ? (
          <div className="text-gray-500 italic">No moves yet</div>
        ) : (
          moves.map((move, index) => (
            <div key={index} className="flex justify-between">
              <span className="font-medium">
                {index + 1}. {move.color} {move.piece_type}
              </span>
              <span className="text-gray-600">
                [{move.from_pos[0]},{move.from_pos[1]}] → [{move.to_pos[0]},{move.to_pos[1]}]
                {move.move_type === 'quantum' && ' ⚛️'}
              </span>
            </div>
          ))
        )}
      </div>
    </Card>
  );
};

function App() {
  const [game, setGame] = useState(null);
  const [selectedSquare, setSelectedSquare] = useState(null);
  const [targetSquare, setTargetSquare] = useState(null);
  const [legalMoves, setLegalMoves] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [showGameModeModal, setShowGameModeModal] = useState(false);
  const [aiThinking, setAiThinking] = useState(false);

  const createNewGame = async (gameConfig = {}) => {
    try {
      setLoading(true);
      setError('');
      setAiThinking(false);
      
      const response = await axios.post(`${API}/game`, gameConfig);
      const newGame = response.data;
      setGame(newGame);
      setSelectedSquare(null);
      setTargetSquare(null);
      setLegalMoves([]);
      
      // If AI is white, make AI's first move
      if (newGame.is_vs_ai && newGame.ai_color === 'white') {
        setTimeout(() => makeAiMove(newGame.id), 1000); // Small delay for better UX
      }
    } catch (err) {
      setError('Failed to create new game: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const fetchLegalMoves = async (gameId) => {
    try {
      const response = await axios.get(`${API}/game/${gameId}/legal-moves`);
      setLegalMoves(response.data.legal_moves);
    } catch (err) {
      console.error('Failed to fetch legal moves:', err);
    }
  };

  const makeAiMove = async (gameId) => {
    try {
      setAiThinking(true);
      setError('');
      
      const response = await axios.post(`${API}/game/${gameId}/ai_move`);
      
      if (response.data.success) {
        setGame(response.data.game);
        await fetchLegalMoves(response.data.game.id);
      } else {
        setError('AI move failed: ' + response.data.message);
      }
    } catch (err) {
      setError('AI move failed: ' + (err.response?.data?.detail || err.message));
    } finally {
      setAiThinking(false);
    }
  };

  const makeMove = async (moveType) => {
    if (!selectedSquare || !targetSquare) return;

    try {
      setLoading(true);
      setError('');
      
      const response = await axios.post(`${API}/game/${game.id}/move`, {
        from_pos: selectedSquare,
        to_pos: targetSquare,
        move_type: moveType
      });
      
      const updatedGame = response.data.game;
      setGame(updatedGame);
      setSelectedSquare(null);
      setTargetSquare(null);
      await fetchLegalMoves(updatedGame.id);
      
      // If it's AI's turn after player move, trigger AI move
      if (updatedGame.is_vs_ai && updatedGame.current_player === updatedGame.ai_color) {
        setTimeout(() => makeAiMove(updatedGame.id), 500); // Small delay for better UX
      }
    } catch (err) {
      setError('Move failed: ' + (err.response?.data?.detail || err.message));
    } finally {
      setLoading(false);
    }
  };

  const measureQuantumPieces = async () => {
    if (!game) return;

    const quantumPositions = [];
    game.board.forEach((row, rowIndex) => {
      row.forEach((square, colIndex) => {
        if (square.quantum_pieces && square.quantum_pieces.length > 0) {
          quantumPositions.push([rowIndex, colIndex]);
        }
      });
    });

    if (quantumPositions.length === 0) return;

    try {
      setLoading(true);
      setError('');
      
      const response = await axios.post(`${API}/game/${game.id}/measure`, {
        positions: quantumPositions
      });
      
      const updatedGame = response.data.game;
      setGame(updatedGame);
      await fetchLegalMoves(updatedGame.id);
      
      // If it's AI's turn after measurement, trigger AI move
      if (updatedGame.is_vs_ai && updatedGame.current_player === updatedGame.ai_color) {
        setTimeout(() => makeAiMove(updatedGame.id), 500); // Small delay for better UX
      }
    } catch (err) {
      setError('Measurement failed: ' + (err.response?.data?.detail || err.message));
    } finally {
      setLoading(false);
    }
  };

  const handleSquareClick = async (row, col) => {
    // Don't allow interactions during AI thinking or if it's AI's turn
    if (aiThinking || (game.is_vs_ai && game.current_player === game.ai_color)) {
      return;
    }
    
    if (!selectedSquare) {
      // Select a square with a piece of the current player
      const square = game.board[row][col];
      if (square.classical_piece && square.classical_piece.color === game.current_player) {
        setSelectedSquare([row, col]);
        return;
      }
    } else if (selectedSquare[0] === row && selectedSquare[1] === col) {
      // Deselect if clicking the same square
      setSelectedSquare(null);
      setTargetSquare(null);
      return;
    } else {
      // Set target square
      setTargetSquare([row, col]);
      return;
    }
  };

  useEffect(() => {
    setShowGameModeModal(true);
  }, []);

  useEffect(() => {
    if (game) {
      fetchLegalMoves(game.id);
    }
  }, [game]);

  if (!game) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center">
        <div className="text-center">
          <div className="text-2xl font-bold text-gray-800 mb-4">Loading Quantum Chess...</div>
          {loading && <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto"></div>}
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-4">
      <div className="max-w-7xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-5xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-purple-600 to-blue-600 mb-2">
            Quantum Chess
          </h1>
          <p className="text-gray-600 text-lg">
            Where classical chess meets quantum mechanics
          </p>
        </div>

        {error && (
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4 text-center">
            {error}
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Chess Board */}
          <div className="lg:col-span-2 flex justify-center">
            <div className="space-y-4">
              <ChessBoard
                game={game}
                onSquareClick={handleSquareClick}
                selectedSquare={selectedSquare}
                legalMoves={legalMoves}
              />
              
              {selectedSquare && targetSquare && (
                <div className="flex space-x-2 justify-center">
                  <Button 
                    onClick={() => makeMove('classical')}
                    disabled={loading}
                    className="px-6"
                  >
                    Make Classical Move
                  </Button>
                  <Button 
                    onClick={() => makeMove('quantum')}
                    disabled={loading || (game.current_player === 'white' ? game.white_superpositions >= 1 : game.black_superpositions >= 1)}
                    variant="outline"
                    className="px-6"
                  >
                    Make Quantum Move
                  </Button>
                </div>
              )}
            </div>
          </div>

          {/* Game Controls & Info */}
          <div className="space-y-6">
            <GameControls
              game={game}
              selectedSquare={selectedSquare}
              onClassicalMove={() => targetSquare && makeMove('classical')}
              onQuantumMove={() => targetSquare && makeMove('quantum')}
              onMeasure={measureQuantumPieces}
              onNewGame={() => setShowGameModeModal(true)}
              aiThinking={aiThinking}
            />
            
            <MoveHistory moves={game.move_history} />
          </div>
        </div>

        <div className="mt-8 text-center text-sm text-gray-500">
          <p>Quantum pieces appear as glowing, semi-transparent duplicates.</p>
          <p>Click a piece, then click destination. Green dots show legal moves.</p>
        </div>
      </div>
    </div>
  );
}

export default App;