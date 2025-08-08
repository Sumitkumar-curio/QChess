import requests
import sys
import json
from datetime import datetime

class QuantumChessAPITester:
    def __init__(self, base_url="https://6e6b16cf-8ba2-4d23-b63a-d161b6bf046f.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0
        self.game_id = None

    def run_test(self, name, method, endpoint, expected_status, data=None, params=None):
        """Run a single API test"""
        url = f"{self.api_url}/{endpoint}" if endpoint else self.api_url
        headers = {'Content-Type': 'application/json'}

        self.tests_run += 1
        print(f"\n🔍 Testing {name}...")
        print(f"   URL: {url}")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, params=params)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=headers)

            print(f"   Response Status: {response.status_code}")
            
            success = response.status_code == expected_status
            if success:
                self.tests_passed += 1
                print(f"✅ Passed - Status: {response.status_code}")
                try:
                    response_data = response.json()
                    print(f"   Response keys: {list(response_data.keys()) if isinstance(response_data, dict) else 'Non-dict response'}")
                    return True, response_data
                except:
                    return True, {}
            else:
                print(f"❌ Failed - Expected {expected_status}, got {response.status_code}")
                try:
                    error_detail = response.json()
                    print(f"   Error: {error_detail}")
                except:
                    print(f"   Error: {response.text}")
                return False, {}

        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            return False, {}

    def test_health_check(self):
        """Test API health check"""
        success, response = self.run_test(
            "API Health Check",
            "GET",
            "",
            200
        )
        return success

    def test_create_game(self):
        """Test game creation"""
        success, response = self.run_test(
            "Create New Game",
            "POST",
            "game",
            200
        )
        if success and 'id' in response:
            self.game_id = response['id']
            print(f"   Game ID: {self.game_id}")
            print(f"   Current Player: {response.get('current_player', 'N/A')}")
            print(f"   Status: {response.get('status', 'N/A')}")
            return True
        return False

    def test_get_game(self):
        """Test getting game state"""
        if not self.game_id:
            print("❌ No game ID available for testing")
            return False
            
        success, response = self.run_test(
            "Get Game State",
            "GET",
            f"game/{self.game_id}",
            200
        )
        if success:
            print(f"   Board size: {len(response.get('board', []))}x{len(response.get('board', [[]])[0]) if response.get('board') else 0}")
            print(f"   White superpositions: {response.get('white_superpositions', 0)}")
            print(f"   Black superpositions: {response.get('black_superpositions', 0)}")
        return success

    def test_get_legal_moves(self):
        """Test getting legal moves"""
        if not self.game_id:
            print("❌ No game ID available for testing")
            return False
            
        success, response = self.run_test(
            "Get Legal Moves",
            "GET",
            f"game/{self.game_id}/legal-moves",
            200
        )
        if success:
            legal_moves = response.get('legal_moves', [])
            print(f"   Legal moves count: {len(legal_moves)}")
            print(f"   Current player: {response.get('current_player', 'N/A')}")
            if legal_moves:
                print(f"   Sample move: {legal_moves[0]}")
        return success

    def test_classical_move(self):
        """Test making a classical move (e2 to e4 - white pawn)"""
        if not self.game_id:
            print("❌ No game ID available for testing")
            return False
            
        # Standard opening move: e2 to e4 (pawn from [6,4] to [4,4])
        move_data = {
            "from_pos": [6, 4],  # e2
            "to_pos": [4, 4],    # e4
            "move_type": "classical"
        }
        
        success, response = self.run_test(
            "Classical Move (e2→e4)",
            "POST",
            f"game/{self.game_id}/move",
            200,
            data=move_data
        )
        if success:
            game = response.get('game', {})
            print(f"   Current player after move: {game.get('current_player', 'N/A')}")
            print(f"   Move history length: {len(game.get('move_history', []))}")
        return success

    def test_black_classical_move(self):
        """Test black's classical move (e7 to e5)"""
        if not self.game_id:
            print("❌ No game ID available for testing")
            return False
            
        # Black responds: e7 to e5 (pawn from [1,4] to [3,4])
        move_data = {
            "from_pos": [1, 4],  # e7
            "to_pos": [3, 4],    # e5
            "move_type": "classical"
        }
        
        success, response = self.run_test(
            "Black Classical Move (e7→e5)",
            "POST",
            f"game/{self.game_id}/move",
            200,
            data=move_data
        )
        if success:
            game = response.get('game', {})
            print(f"   Current player after move: {game.get('current_player', 'N/A')}")
            print(f"   Move history length: {len(game.get('move_history', []))}")
        return success

    def test_quantum_move(self):
        """Test making a quantum move (d2 to d4 - white pawn quantum)"""
        if not self.game_id:
            print("❌ No game ID available for testing")
            return False
            
        # Quantum move: d2 to d4 (pawn from [6,3] to [4,3])
        move_data = {
            "from_pos": [6, 3],  # d2
            "to_pos": [4, 3],    # d4
            "move_type": "quantum"
        }
        
        success, response = self.run_test(
            "Quantum Move (d2→d4)",
            "POST",
            f"game/{self.game_id}/move",
            200,
            data=move_data
        )
        if success:
            game = response.get('game', {})
            print(f"   White superpositions: {game.get('white_superpositions', 0)}")
            print(f"   Current player after move: {game.get('current_player', 'N/A')}")
        return success

    def test_quantum_move_limit(self):
        """Test quantum move limit (should fail after 1 superposition)"""
        if not self.game_id:
            print("❌ No game ID available for testing")
            return False
            
        # Try another quantum move for white (should fail)
        move_data = {
            "from_pos": [6, 2],  # c2
            "to_pos": [4, 2],    # c4
            "move_type": "quantum"
        }
        
        success, response = self.run_test(
            "Quantum Move Limit Test (should fail)",
            "POST",
            f"game/{self.game_id}/move",
            400,  # Should fail with 400
            data=move_data
        )
        return success

    def test_measure_quantum_pieces(self):
        """Test measuring quantum pieces"""
        if not self.game_id:
            print("❌ No game ID available for testing")
            return False
            
        # Measure all quantum positions
        measure_data = {
            "positions": [[6, 3], [4, 3]]  # Both positions of the quantum superposition
        }
        
        success, response = self.run_test(
            "Measure Quantum Pieces",
            "POST",
            f"game/{self.game_id}/measure",
            200,
            data=measure_data
        )
        if success:
            game = response.get('game', {})
            print(f"   White superpositions after measure: {game.get('white_superpositions', 0)}")
        return success

    def test_invalid_game_id(self):
        """Test with invalid game ID"""
        success, response = self.run_test(
            "Invalid Game ID",
            "GET",
            "game/invalid-id-123",
            404
        )
        return success

    def test_invalid_move(self):
        """Test invalid move"""
        if not self.game_id:
            print("❌ No game ID available for testing")
            return False
            
        # Try to move a piece that doesn't exist or invalid move
        move_data = {
            "from_pos": [0, 0],  # Black rook
            "to_pos": [7, 7],    # Invalid move
            "move_type": "classical"
        }
        
        success, response = self.run_test(
            "Invalid Move Test",
            "POST",
            f"game/{self.game_id}/move",
            400,
            data=move_data
        )
        return success

def main():
    print("🚀 Starting Quantum Chess API Tests")
    print("=" * 50)
    
    tester = QuantumChessAPITester()
    
    # Run all tests in sequence
    tests = [
        tester.test_health_check,
        tester.test_create_game,
        tester.test_get_game,
        tester.test_get_legal_moves,
        tester.test_classical_move,
        tester.test_black_classical_move,
        tester.test_quantum_move,
        tester.test_quantum_move_limit,
        tester.test_measure_quantum_pieces,
        tester.test_invalid_game_id,
        tester.test_invalid_move
    ]
    
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"❌ Test failed with exception: {str(e)}")
    
    # Print final results
    print("\n" + "=" * 50)
    print(f"📊 Final Results: {tester.tests_passed}/{tester.tests_run} tests passed")
    
    if tester.tests_passed == tester.tests_run:
        print("🎉 All tests passed! Backend API is working correctly.")
        return 0
    else:
        print(f"⚠️  {tester.tests_run - tester.tests_passed} tests failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())