"""Quick debug script for orchestrator."""
import sys
sys.path.insert(0, '.')

from tactiq_ai.agents.tactical_analyst import TacticalAnalystAgent
from tactiq_ai.agents.keras_inference_agent import KerasInferenceAgent
from tactiq_ai.agents.player_analyst import PlayerAnalystAgent
from tactiq_ai.agents.orchestrator import TactiqOrchestrator

events = [
    {'event_type': 'pass', 'team_id': 'home', 'player_id': 'p1', 'player_name': 'Player 1', 'position': 'MF',
     'start_x': 50, 'start_y': 50, 'end_x': 80, 'end_y': 50,
     'timestamp_seconds': 10, 'match_id': 'test', 'completed': True},
    {'event_type': 'pass', 'team_id': 'away', 'player_id': 'p2', 'player_name': 'Player 2', 'position': 'MF',
     'start_x': 30, 'start_y': 70, 'end_x': 10, 'end_y': 70,
     'timestamp_seconds': 12, 'match_id': 'test', 'completed': True},
    {'event_type': 'pass', 'team_id': 'home', 'player_id': 'p3', 'player_name': 'Player 3', 'position': 'FW',
     'start_x': 80, 'start_y': 50, 'end_x': 100, 'end_y': 50,
     'timestamp_seconds': 15, 'match_id': 'test', 'completed': True},
]

print('1. Keras inference:')
inference = KerasInferenceAgent()
inference_result = inference.run(events)
print(f'   Type: {type(inference_result)}')
print(f'   Length: {len(inference_result)}')
if inference_result:
    print(f'   First: {inference_result[0]}')

print()
print('2. Player analyst:')
player = PlayerAnalystAgent()
player_result = player.analyze(events)
print(f'   Type: {type(player_result)}')
print(f'   Length: {len(player_result)}')
if player_result:
    first = player_result[0]
    print(f'   First element type: {type(first)}')
    print(f'   First has .get: {hasattr(first, "get")}')

print()
print('3. Run orchestrator:')
orchestrator = TactiqOrchestrator()
try:
    result = orchestrator.run(events)
    print('   Success!')
except Exception as e:
    print(f'   Error: {type(e).__name__}: {e}')
    import traceback
    traceback.print_exc()