import os
import sys
import time
import argparse
import importlib
from dataclasses import dataclass
from typing import Optional, List, Dict

import numpy as np
import torch
import torch.nn as nn

from cr_engine import CREngine
from cr_gym_env import CRGymEnv


# ------------------------------
# Minimal default Config
# ------------------------------
@dataclass
class Config:
  manager_hidden: int = 128
  worker_hidden: int = 128
  intent_dim: int = 8
  manager_intent_horizon: int = 4
  worker_max_iter: int = 3
  worker_act_eps: float = 0.05
  ponder_cost: float = 0.01


def clear_screen():
  if os.name == 'nt':
    os.system('cls')
  else:
    # ANSI clear
    sys.stdout.write('\033[2J\033[H')
    sys.stdout.flush()


class AgentPolicyAdapterForEngine:
  """Adapter that lets CREngine query a hierarchical agent.

  It converts CREngine observations to CRGymEnv-style observations by temporarily
  sharing the same engine inside a CRGymEnv instance, then calls the agent.
  """

  def __init__(self, agent: nn.Module, device: torch.device, shared_engine: CREngine, max_entities: int = 64):
    self.agent = agent
    self.device = device
    self._last_decision: Optional[Dict[str, int]] = None

    # CRGymEnv wrapper to build entity/tower/hand encodings from the engine
    self._enc_env = CRGymEnv(max_entities=max_entities, perspective=0)
    # Share the provided engine so observations reflect the same state
    self._enc_env.engine = shared_engine

    # Initialize agent recurrent/intents if available
    if hasattr(agent, 'manager_h'):
      agent.manager_h = None
    if hasattr(agent, 'worker_h'):
      agent.worker_h = None
    if hasattr(agent, 'cur_intent'):
      agent.cur_intent = None
    if hasattr(agent, 'steps_left'):
      agent.steps_left = 0

  def _infer_action_triplet(self, team: int) -> Dict[str, int]:
    # Build CRGymEnv-format observation for the requested team
    self._enc_env.perspective = team
    obs = self._enc_env._obs()  # {'entities','entities_mask','towers','hand','action_masks'}

    entities_t = torch.from_numpy(obs['entities'].astype(np.float32)).unsqueeze(0).to(self.device)
    mask_t = torch.from_numpy(obs['entities_mask'].astype(bool)).unsqueeze(0).to(self.device)
    towers_t = torch.from_numpy(obs['towers'].astype(np.float32)).unsqueeze(0).to(self.device)
    hand_t = torch.from_numpy(obs['hand'].astype(np.float32)).unsqueeze(0).to(self.device)
    emb = self.agent.encode(entities_t, mask_t.bool(), towers_t, hand_t)

    card_mask_t = torch.from_numpy(obs['action_masks']['card']).view(1, -1).to(self.device)
    lane_mask_t = torch.from_numpy(obs['action_masks']['lane']).view(1, -1).to(self.device)
    place_mask_t = torch.from_numpy(obs['action_masks']['place']).view(1, -1).to(self.device)
    masks = {'card': card_mask_t, 'lane': lane_mask_t, 'place': place_mask_t}

    # Support either new API step(...) or legacy act(...)
    self.agent.eval()
    with torch.no_grad():
      if hasattr(self.agent, 'step'):
        actions, _, _, _ = self.agent.step(emb, masks)
      else:
        # Legacy: act(emb, intent_onehot, masks, h_worker)
        intent_zero = torch.zeros((1, getattr(self.agent, 'intent_dim', 1)), device=self.device)
        actions, _, _, _, _ = self.agent.act(emb, intent_zero, masks, None)

    return {
      'card': int(actions['card'].item()),
      'lane': int(actions['lane'].item()),
      'place': int(actions['place'].item()),
    }

  # CREngine policy interface
  def select_card_action(self, obs_engine: Dict, legal_card_actions: List[int]) -> int:
    # Compute all three at once, cache for lane/place queries
    # team perspective is derived from which obs is passed by CREngine; here we detect by comparing towers
    # If obs contains my towers at top row, assume team 0; else 1. Simpler: try both and trust masks.
    # We can infer team by comparing resource to engine players, but we don't have direct handle here.
    # Use team 1 for opponent calls from engine.human_vs_agent where bot is always the opposite team.
    # The caller (play loop) will pass the correct team in our higher-level wrapper.
    # Here, we just compute once via team placeholder updated by wrapper.
    team = getattr(self, '_current_team', 1)
    decision = self._infer_action_triplet(team)

    # Enforce legality: if chosen card illegal, fallback to skip (2) or first legal
    if decision['card'] not in legal_card_actions:
      decision['card'] = 2 if 2 in legal_card_actions else (legal_card_actions[0] if legal_card_actions else 2)

    self._last_decision = decision
    return decision['card']

  def select_lane_action(self, obs_engine: Dict, legal_lane_actions: List[int]) -> int:
    if self._last_decision is None:
      self._last_decision = {'card': 2, 'lane': 0, 'place': 0}
    lane = self._last_decision['lane']
    if lane not in legal_lane_actions:
      lane = legal_lane_actions[0]
    return int(lane)

  def select_place_action(self, obs_engine: Dict, legal_place_actions: List[int]) -> int:
    if self._last_decision is None:
      self._last_decision = {'card': 2, 'lane': 0, 'place': 0}
    place = self._last_decision['place']
    if place not in legal_place_actions:
      place = legal_place_actions[0]
    return int(place)


def play_human_vs_agent(agent: nn.Module,
                        device: torch.device,
                        human_team: int = 0,
                        max_entities: int = 64,
                        refresh: bool = True,
                        sleep_s: float = 0.0):
  engine = CREngine()
  adapter = AgentPolicyAdapterForEngine(agent=agent, device=device, shared_engine=engine, max_entities=max_entities)

  engine.reset()
  turn = 0
  while not engine.is_done():
    persp = human_team
    if refresh:
      clear_screen()
    print(engine.render_ascii(persp))
    print("-" * 32)

    # Human input
    try:
      legal_cards = engine.legal_card_actions(persp)
      print(f"Legal cards: {legal_cards} (0=front,1=pre-front,2=skip)")
      a_card = int(input("Card> ").strip())
    except Exception:
      a_card = 2
    a_card = a_card if a_card in engine.legal_card_actions(persp) else 2

    a_lane = 0
    a_place = 0
    if a_card in [0, 1]:
      try:
        a_lane = int(input("Lane (0=left,1=right)> ").strip())
      except Exception:
        a_lane = 0
      a_lane = a_lane if a_lane in engine.legal_lane_actions(persp) else 0
      try:
        a_place = int(input("Place (0=near,1=far)> ").strip())
      except Exception:
        a_place = 0
      a_place = a_place if a_place in engine.legal_place_actions(persp) else 0

    # Bot (opponent)
    opp_team = 1 - human_team
    adapter._current_team = opp_team  # hint team to adapter
    opp_obs = engine.get_observation(opp_team)
    opp_card = adapter.select_card_action(opp_obs, engine.legal_card_actions(opp_team))
    opp_lane = adapter.select_lane_action(opp_obs, engine.legal_lane_actions(opp_team))
    opp_place = adapter.select_place_action(opp_obs, engine.legal_place_actions(opp_team))

    # Step
    if human_team == 0:
      engine.step(a_card, a_lane, a_place, opp_card, opp_lane, opp_place)
    else:
      engine.step(opp_card, opp_lane, opp_place, a_card, a_lane, a_place)

    turn += 1
    if sleep_s > 0:
      time.sleep(sleep_s)

  if refresh:
    clear_screen()
  print(engine.render_ascii(human_team))
  winner = engine.compute_tiebreak_winner()
  if winner == -1:
    print("Result: Draw")
  else:
    print("Result: You win" if winner == human_team else "Result: Bot wins")


def load_agent_from_args(args, device: torch.device):
  # Dynamic import of agent class; fallback to informative error
  if args.agent_module is None:
    raise RuntimeError("--agent-module is required (e.g., 'your_module.path')")
  module = importlib.import_module(args.agent_module)
  cls_name = args.agent_class or 'HRMAgentEntitiesMulti'
  AgentCls = getattr(module, cls_name)

  # Probe dims using a temporary engine+encoder
  temp_engine = CREngine()
  env = CRGymEnv(max_entities=args.max_entities, perspective=0)
  env.engine = temp_engine
  obs0 = env._obs_for(0)
  entity_dim = int(obs0['entities'].shape[-1])
  towers_dim = int(obs0['towers'].shape[-1])

  cfg = Config(manager_hidden=args.manager_hidden,
               worker_hidden=args.worker_hidden,
               intent_dim=args.intent_dim,
               manager_intent_horizon=args.intent_horizon,
               worker_max_iter=args.worker_max_iter,
               worker_act_eps=args.worker_act_eps,
               ponder_cost=args.ponder_cost)

  agent = AgentCls(cfg, entity_dim=entity_dim, towers_dim=towers_dim, model_dim=args.model_dim).to(device)

  if args.checkpoint:
    state = torch.load(args.checkpoint, map_location=device)
    # allow state wrappers (e.g., {'model': sd})
    sd = state.get('model', state) if isinstance(state, dict) else state
    agent.load_state_dict(sd, strict=False)
  agent.eval()
  return agent


def parse_args():
  p = argparse.ArgumentParser(description='Play human vs HRM agent (CREngine) with screen refresh.')
  p.add_argument('--agent-module', type=str, required=True, help='Python module path containing the agent class')
  p.add_argument('--agent-class', type=str, default='HRMAgentEntitiesMulti', help='Agent class name in the module')
  p.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint (state_dict)')
  p.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'])
  p.add_argument('--human-team', type=int, default=0, choices=[0, 1])
  p.add_argument('--max-entities', type=int, default=64)
  p.add_argument('--model-dim', type=int, default=128)

  # Config overrides
  p.add_argument('--manager-hidden', type=int, default=128)
  p.add_argument('--worker-hidden', type=int, default=128)
  p.add_argument('--intent-dim', type=int, default=8)
  p.add_argument('--intent-horizon', type=int, default=4)
  p.add_argument('--worker-max-iter', type=int, default=3)
  p.add_argument('--worker-act-eps', type=float, default=0.05)
  p.add_argument('--ponder-cost', type=float, default=0.01)

  p.add_argument('--no-refresh', action='store_true', help='Disable screen clearing')
  p.add_argument('--sleep', type=float, default=0.0, help='Sleep seconds after each step (for readability)')
  return p.parse_args()


def main():
  args = parse_args()
  if args.device == 'auto':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  else:
    device = torch.device(args.device)

  agent = load_agent_from_args(args, device)
  play_human_vs_agent(agent=agent,
                      device=device,
                      human_team=args.human_team,
                      max_entities=args.max_entities,
                      refresh=(not args.no_refresh),
                      sleep_s=args.sleep)


if __name__ == '__main__':
  main()
