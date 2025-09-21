import numpy as np
from typing import Dict, Tuple, List, Optional

from cr_engine import CREngine, RandomPolicy, Team, Unit


class CRGymEnv:
  """Single-agent wrapper around CREngine with entity embeddings and action masks.

	Observation dict:
	- entities: float32 [max_entities, feat_dim] where feat = [x_norm, y_norm, onehot_4, hp_pct, is_friendly]
	- entities_mask: bool [max_entities]
	- towers: float32 [6] -> [my_K,L,R, opp_K,L,R] normalized to [0,1]
	- hand: float32 [2, 5] -> per slot [onehot_4, cost_pct]
	- action_masks: dict with 'card' [3], 'lane' [2], 'place' [2] (int8 {0,1})

	Action dict:
	- { 'card': int in {0,1,2}, 'lane': int in {0,1}, 'place': int in {0,1} }
  """

  def __init__(self, max_entities: int = 64, perspective: Team = 0, opponent_policy: Optional[object] = None):
    self.engine = CREngine()
    self.max_entities = max_entities
    self.perspective = perspective
    self.opponent_policy = opponent_policy or RandomPolicy()
    self.type_to_idx = {"Melee": 0, "Ranged": 1, "Tank": 2, "Fast": 3}
    self.feat_dim = 2 + 4 + 1 + 1

  def reset(self) -> Dict:
    self.engine.reset()
    return self._obs()

  def step(self, action: Dict[str, int]) -> Tuple[Dict, float, bool, Dict]:
    card = int(action.get('card', 2))
    lane = int(action.get('lane', 0))
    place = int(action.get('place', 0))
    # Build opponent observation in CRGymEnv format
    opp_obs = self._obs_for(1 - self.perspective)
    opp_card = self.opponent_policy.select_card_action(opp_obs, self.engine.legal_card_actions(1 - self.perspective))
    opp_lane = self.opponent_policy.select_lane_action(opp_obs, self.engine.legal_lane_actions(1 - self.perspective))
    opp_place = self.opponent_policy.select_place_action(opp_obs, [0, 1])

    if self.perspective == 0:
      (_, _), (r0, r1), done, info = self.engine.step(card, lane, place, opp_card, opp_lane, opp_place)
      reward = float(r0)
    else:
      (_, _), (r0, r1), done, info = self.engine.step(opp_card, opp_lane, opp_place, card, lane, place)
      reward = float(r1)
    return self._obs(), reward, bool(done), info

  def _obs(self) -> Dict:
    entities, mask = self._encode_entities()
    towers = self._encode_towers()
    hand = self._encode_hand()
    masks = self._action_masks()
    return {
      'entities': entities,
      'entities_mask': mask,
      'towers': towers,
      'hand': hand,
      'action_masks': masks,
    }

  def _obs_for(self, team: Team) -> Dict:
    # Temporarily switch perspective to build team-specific observation
    cur = self.perspective
    try:
      self.perspective = team
      return self._obs()
    finally:
      self.perspective = cur

  def _normalize_xy(self, row: int, col: int) -> Tuple[float, float]:
    H, W = self.engine.cfg.board_h, self.engine.cfg.board_w
    x = (2.0 * col / (W - 1)) - 1.0
    y = (2.0 * row / (H - 1)) - 1.0
    return x, y

  def _encode_entities(self) -> Tuple[np.ndarray, np.ndarray]:
    features: List[List[float]] = []
    for u in self.engine.units:
      x, y = self._normalize_xy(u.row, u.col)
      onehot = [0.0, 0.0, 0.0, 0.0]
      onehot[self.type_to_idx[u.unit_type.name]] = 1.0
      hp_pct = float(u.hp) / float(u.unit_type.hp) if u.unit_type.hp > 0 else 0.0
      is_friendly = 1.0 if u.team == self.perspective else 0.0
      feat = [x, y] + onehot + [hp_pct, is_friendly]
      features.append(feat)
    mask = np.zeros((self.max_entities, ), dtype=np.bool_)
    arr = np.zeros((self.max_entities, self.feat_dim), dtype=np.float32)
    count = min(len(features), self.max_entities)
    if count > 0:
      arr[:count, :] = np.asarray(features[:count], dtype=np.float32)
      mask[:count] = True
    return arr, mask

  def _encode_towers(self) -> np.ndarray:
    cfg = self.engine.cfg
    max_hp = float(cfg.initial_tower_hp)
    my = self.engine.towers[self.perspective]
    opp = self.engine.towers[1 - self.perspective]
    vec = [
      float(my['king']) / max_hp,
      float(my['left']) / max_hp,
      float(my['right']) / max_hp,
      float(opp['king']) / max_hp,
      float(opp['left']) / max_hp,
      float(opp['right']) / max_hp,
    ]
    return np.asarray(vec, dtype=np.float32)

  def _encode_hand(self) -> np.ndarray:
    # Two slots: deck_queue[0], deck_queue[1]
    ps = self.engine.players[self.perspective]
    max_hp = float(self.engine.cfg.initial_tower_hp)
    slots = []
    for slot in [0, 1]:
      onehot = [0.0, 0.0, 0.0, 0.0]
      cost_pct = 0.0
      if slot < len(ps.deck_queue):
        idx = ps.deck_queue[slot]
        card = ps.deck[idx]
        onehot[self.type_to_idx[card.unit_type.name]] = 1.0
        cost_pct = float(card.cost) / 10.0
      slots.append(onehot + [cost_pct])
    return np.asarray(slots, dtype=np.float32)

  def _action_masks(self) -> Dict[str, np.ndarray]:
    card_legal = self.engine.legal_card_actions(self.perspective)
    card_mask = np.zeros((3, ), dtype=np.int8)
    for a in card_legal:
      if a in (0, 1, 2):
        card_mask[a] = 1
    lane_mask = np.ones((2, ), dtype=np.int8)
    place_mask = np.ones((2, ), dtype=np.int8)
    return {'card': card_mask, 'lane': lane_mask, 'place': place_mask}
