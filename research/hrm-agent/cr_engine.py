import random
import sys
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np

Team = int  # 0 or 1


@dataclass
class CRConfig:
  board_w: int = 9
  board_h: int = 7
  river_row: int = 3
  max_turns: int = 200
  initial_tower_hp: int = 20
  spawn_row_offset: int = 1

  # Lanes mapped to fixed columns
  left_lane_col: int = 2
  right_lane_col: int = 6

  # Elixir/resource system
  max_resource: int = 10
  resource_per_turn: int = 1
  starting_resource: int = 4

  # Unit archetypes
  melee_hp: int = 5
  melee_damage: int = 2
  melee_range: int = 1
  melee_speed: int = 1

  ranged_hp: int = 5
  ranged_damage: int = 2
  ranged_range: int = 3
  ranged_speed: int = 1

  tank_hp: int = 12
  tank_damage: int = 3
  tank_range: int = 1
  tank_speed: int = 1

  fast_hp: int = 3
  fast_damage: int = 2
  fast_range: int = 1
  fast_speed: int = 2

  # Reward shaping
  damage_weight: float = 0.1  # small per-step weight on tower damage
  win_reward: float = 2.0  # strong terminal bonus/penalty
  reward_multiplier: float = 1e-3


@dataclass
class UnitType:
  name: str
  hp: int
  damage: int
  range_cells: int
  speed_cells: int
  symbol: str  # base symbol, e.g., 'M','R','T','F'


@dataclass
class Unit:
  unit_type: UnitType
  team: Team
  row: int
  col: int
  hp: int
  spawn_cooldown: int = 0

  def is_alive(self) -> bool:
    return self.hp > 0


@dataclass
class Card:
  name: str
  unit_type: UnitType
  cost: int


@dataclass
class PlayerState:
  deck: List[Card]
  deck_queue: List[int]
  resource: int


class CREngine:
  """Simplified Clash Royale-like engine with two lanes and fixed spawn columns.

	- Two players with identical 4-card decks.
	- Each turn, each player may play at most one card from a 2-card hand or skip.
	- Units move along their lane (vertical only) toward the opponent side.
	- Deterministic targeting and movement stop rules:
	  * Ranged: can target across lanes; if a target is in range, it holds position and shoots.
	  * Tank: ignores enemy units; targets towers only; moves through units; stops only to shoot towers in range.
	  * Others (Melee, Fast): if a legal target is in range, they fight in place; otherwise move forward.
	- Tower preference: prefer same-lane princess tower; if destroyed, engage king tower ("king unlock").
	- Reward is damage dealt to enemy towers minus damage taken by own towers.
	- Decisions are factorized into card selection (hand slot or skip) and lane selection (left/right).
  """

  def __init__(self, cfg: Optional[CRConfig] = None):
    self.cfg = cfg or CRConfig()
    self.turn = 0
    self.units: List[Unit] = []
    self.towers: Dict[Team, Dict[str, int]] = {
      0: {
        "king": self.cfg.initial_tower_hp,
        "left": self.cfg.initial_tower_hp,
        "right": self.cfg.initial_tower_hp
      },
      1: {
        "king": self.cfg.initial_tower_hp,
        "left": self.cfg.initial_tower_hp,
        "right": self.cfg.initial_tower_hp
      },
    }
    self.tower_positions: Dict[Team, Dict[str, Tuple[int, int]]] = self._init_tower_positions()
    self.unit_types = self._init_unit_types()
    self.base_deck = self._init_deck()
    self.players: Dict[Team, PlayerState] = {}
    self._init_players()

  def _init_tower_positions(self) -> Dict[Team, Dict[str, Tuple[int, int]]]:
    mid = self.cfg.board_w // 2
    positions = {
      0: {
        "king": (0, mid),
        "left": (0, self.cfg.left_lane_col),
        "right": (0, self.cfg.right_lane_col)
      },
      1: {
        "king": (self.cfg.board_h - 1, mid),
        "left": (self.cfg.board_h - 1, self.cfg.left_lane_col),
        "right": (self.cfg.board_h - 1, self.cfg.right_lane_col)
      },
    }
    return positions

  def _init_unit_types(self) -> Dict[str, UnitType]:
    return {
      "Melee":
      UnitType("Melee", self.cfg.melee_hp, self.cfg.melee_damage, self.cfg.melee_range, self.cfg.melee_speed, "M"),
      "Ranged":
      UnitType("Ranged", self.cfg.ranged_hp, self.cfg.ranged_damage, self.cfg.ranged_range, self.cfg.ranged_speed, "R"),
      "Tank":
      UnitType("Tank", self.cfg.tank_hp, self.cfg.tank_damage, self.cfg.tank_range, self.cfg.tank_speed, "T"),
      "Fast":
      UnitType("Fast", self.cfg.fast_hp, self.cfg.fast_damage, self.cfg.fast_range, self.cfg.fast_speed, "F"),
    }

  def _init_deck(self) -> List[Card]:
    ut = self.unit_types
    # Deck of 4 cards with costs; lane is chosen independently at play time
    return [
      Card(name="Melee", unit_type=ut["Melee"], cost=3),
      Card(name="Ranged", unit_type=ut["Ranged"], cost=4),
      Card(name="Tank", unit_type=ut["Tank"], cost=5),
      Card(name="Fast", unit_type=ut["Fast"], cost=2),
    ]

  def _init_players(self):
    for team in [0, 1]:
      deck = list(self.base_deck)
      deck_queue = list(range(len(deck)))
      random.shuffle(deck_queue)
      self.players[team] = PlayerState(
        deck=deck,
        deck_queue=deck_queue,
        resource=self.cfg.starting_resource,
      )

  def reset(self):
    self.turn = 0
    self.units.clear()
    for t in [0, 1]:
      for k in self.towers[t].keys():
        self.towers[t][k] = self.cfg.initial_tower_hp
    self._init_players()
    return self.get_observation(0), self.get_observation(1)

  def legal_card_actions(self, team: Team) -> List[int]:
    # 0: play deck_queue[0] (front), 1: play deck_queue[1] (pre-front), 2: skip
    actions: List[int] = [2]
    ps = self.players[team]
    for slot in [0, 1]:
      if slot >= len(ps.deck_queue):
        continue
      idx = ps.deck_queue[slot]
      card = ps.deck[idx]
      if ps.resource >= card.cost:
        actions.append(slot)
    actions = sorted(set(actions))
    return actions

  def legal_lane_actions(self, team: Team) -> List[int]:
    # Lane: 0=left, 1=right
    return [0, 1]

  def legal_place_actions(self, team: Team) -> List[int]:
    # Placement: 0=near (in front of towers), 1=far (cell before river on chosen lane)
    return [0, 1]

  def _lane_col(self, lane_action: int) -> int:
    return self.cfg.left_lane_col if lane_action == 0 else self.cfg.right_lane_col

  def _other_lane_col(self, col: int) -> int:
    return self.cfg.right_lane_col if col == self.cfg.left_lane_col else self.cfg.left_lane_col

  def _is_in_attack_range(self, u: Unit, target_row: int, target_col: int) -> bool:
    # Special cross-lane rule for ranged: can shoot the opposite lane column at same row or +-1 rows
    if u.unit_type.name == "Ranged":
      opp_col = self._other_lane_col(u.col)
      if target_col == opp_col and abs(target_row - u.row) <= 1:
        return True
    # Default Manhattan distance check
    return (abs(u.row - target_row) + abs(u.col - target_col)) <= u.unit_type.range_cells

  def _spawn_unit(self, team: Team, card: Card, lane_action: int, place_action: int):
    lane_col = self._lane_col(lane_action)
    if place_action == 0:  # near
      tower_row = self.tower_positions[team]["king"][0]
      spawn_row = tower_row + (1 if team == 0 else -1)
    else:  # far
      spawn_row = self.cfg.river_row - 1 if team == 0 else self.cfg.river_row + 1
    u = Unit(unit_type=card.unit_type,
             team=team,
             row=int(np.clip(spawn_row, 0, self.cfg.board_h - 1)),
             col=lane_col,
             hp=card.unit_type.hp,
             spawn_cooldown=1)
    self.units.append(u)

  def _draw_replacement(self, team: Team, hand_slot: int):
    ps = self.players[team]
    # Pop the chosen slot from the front pair and push its index to the back
    if hand_slot < 0 or hand_slot >= len(ps.deck_queue):
      return
    played_idx = ps.deck_queue.pop(hand_slot)
    ps.deck_queue.append(played_idx)

  def _unit_lane_for_col(self, col: int, team: Team) -> str:
    # Determine closest lane name for a column with team-symmetric tie-break
    dl = abs(col - self.cfg.left_lane_col)
    dr = abs(col - self.cfg.right_lane_col)
    if dl < dr:
      return "left"
    if dr < dl:
      return "right"
    # Tie at center column: break symmetrically by team to avoid global left bias
    return "left" if team == 0 else "right"

  def _preferred_tower_target(self, opponent: Team, unit_col: int,
                              unit_team: Team) -> Tuple[Optional[Tuple[Team, str]], Optional[Tuple[Team, str]]]:
    """Return (preferred princess, fallback king) keys for the opponent."""
    lane = self._unit_lane_for_col(unit_col, unit_team)
    princess = (opponent, "left") if lane == "left" else (opponent, "right")
    king = (opponent, "king")
    return princess, king

  def _has_target_in_range(self, u: Unit) -> bool:
    opponent = 1 - u.team
    # Tank: only towers
    consider_units = (u.unit_type.name != "Tank")
    if consider_units:
      for v in self.units:
        if not v.is_alive() or v.team == u.team:
          continue
        if self._is_in_attack_range(u, v.row, v.col):
          return True
    princess_key, king_key = self._preferred_tower_target(opponent, u.col, u.team)
    for (team, name) in [princess_key, king_key]:
      pos = self.tower_positions[team][name]
      if self.towers[team][name] > 0 and self._is_in_attack_range(u, pos[0], pos[1]):
        return True
    return False

  def _advance_units(self):
    for u in self.units:
      if not u.is_alive():
        continue
      if u.spawn_cooldown > 0:
        u.spawn_cooldown -= 1
        continue

      if u.unit_type.name == "Tank":
        opponent = 1 - u.team
        princess_key, king_key = self._preferred_tower_target(opponent, u.col, u.team)
        stop = False
        for (team, name) in [princess_key, king_key]:
          pos = self.tower_positions[team][name]
          if self.towers[team][name] > 0 and self._is_in_attack_range(u, pos[0], pos[1]):
            stop = True
            break
      else:
        stop = self._has_target_in_range(u)
      if stop:
        continue

      opponent = 1 - u.team
      princess_key, king_key = self._preferred_tower_target(opponent, u.col, u.team)
      princess_alive = self.towers[princess_key[0]][princess_key[1]] > 0
      # Desired column: lane princess tower column if alive; otherwise king column
      desired_col = self.tower_positions[opponent][
        princess_key[1]][1] if princess_alive else self.tower_positions[opponent]["king"][1]

      dir_row = 1 if u.team == 0 else -1
      for _ in range(u.unit_type.speed_cells):
        # horizontal drift toward desired column when moving
        if u.col < desired_col:
          u.col += 1
        elif u.col > desired_col:
          u.col -= 1
        u.col = int(np.clip(u.col, 0, self.cfg.board_w - 1))
        # then vertical step
        u.row = int(np.clip(u.row + dir_row, 0, self.cfg.board_h - 1))

  def _collect_targets(self) -> Tuple[Dict[int, int], Dict[Tuple[Team, str], int], List[int]]:
    dmg_units: Dict[int, int] = {}
    dmg_towers: Dict[Tuple[Team, str], int] = {}
    attacking_king: List[int] = []

    for i, u in enumerate(self.units):
      if not u.is_alive():
        continue
      if u.spawn_cooldown > 0:
        continue
      opponent = 1 - u.team

      if u.unit_type.name == "Tank":
        princess_key, king_key = self._preferred_tower_target(opponent, u.col, u.team)
        chosen: Optional[Tuple[Team, str]] = None
        for key in [princess_key, king_key]:
          team, name = key
          if self.towers[team][name] <= 0:
            continue
          pos = self.tower_positions[team][name]
          if self._is_in_attack_range(u, pos[0], pos[1]):
            chosen = key
            break
        if chosen is not None:
          dmg_towers[chosen] = dmg_towers.get(chosen, 0) + u.unit_type.damage
          if chosen[1] == "king":
            attacking_king.append(i)
        continue

      best_unit_idx: Optional[int] = None
      best_unit_metric: Optional[Tuple[int, int, int, int, str, int]] = None
      for j, v in enumerate(self.units):
        if not v.is_alive() or v.team == u.team:
          continue
        if self._is_in_attack_range(u, v.row, v.col):
          drow = abs(u.row - v.row)
          dcol = abs(u.col - v.col)
          dist = drow + dcol
          # Team-symmetric tie-breakers to avoid global left/right bias
          mid_col = self.cfg.board_w // 2
          metric = (
            dist,
            drow,
            abs(v.col - mid_col),
            (v.col if u.team == 0 else -v.col),
            v.unit_type.name,
            j,
          )
          if best_unit_metric is None or metric < best_unit_metric:
            best_unit_metric = metric
            best_unit_idx = j
      if best_unit_idx is not None:
        dmg_units[best_unit_idx] = dmg_units.get(best_unit_idx, 0) + u.unit_type.damage
        continue

      princess_key, king_key = self._preferred_tower_target(opponent, u.col, u.team)
      for key in [princess_key, king_key]:
        team, name = key
        if self.towers[team][name] <= 0:
          continue
        pos = self.tower_positions[team][name]
        if self._is_in_attack_range(u, pos[0], pos[1]):
          dmg_towers[key] = dmg_towers.get(key, 0) + u.unit_type.damage
          if name == "king":
            attacking_king.append(i)
          break

    return dmg_units, dmg_towers, attacking_king

  def _apply_lead_damage(self):
    # 1 damage per turn to the foremost unit on enemy side for each team and lane where the opponent's lane tower is alive
    for team in [0, 1]:
      opponent = 1 - team
      for lane_name, lane_col in [("left", self.cfg.left_lane_col), ("right", self.cfg.right_lane_col)]:
        if self.towers[opponent][lane_name] <= 0:
          continue
        # Candidates: units of 'team' beyond river and aligned to this lane
        def is_in_lane(u: Unit) -> bool:
          return self._unit_lane_for_col(u.col, u.team) == lane_name

        if team == 0:
          candidates = [u for u in self.units if u.team == team and u.row > self.cfg.river_row and is_in_lane(u)]
          if not candidates:
            continue
          lead = max(candidates, key=lambda u: u.row)
        else:
          candidates = [u for u in self.units if u.team == team and u.row < self.cfg.river_row and is_in_lane(u)]
          if not candidates:
            continue
          lead = min(candidates, key=lambda u: u.row)
        lead.hp -= 1

  def _apply_damage(self, dmg_units: Dict[int, int], dmg_towers: Dict[Tuple[Team, str], int]):
    for idx, dmg in dmg_units.items():
      if 0 <= idx < len(self.units) and self.units[idx].is_alive():
        self.units[idx].hp -= dmg
    for (team, name), dmg in dmg_towers.items():
      if self.towers[team][name] > 0:
        self.towers[team][name] -= dmg
    self.units = [u for u in self.units if u.is_alive()]

  def _apply_king_counter_damage(self, attacker_indices: List[int]):
    for idx in attacker_indices:
      if 0 <= idx < len(self.units) and self.units[idx].is_alive():
        self.units[idx].hp -= 1

  def _compute_reward(self, prev_towers: Dict[Team, Dict[str, int]]) -> Tuple[float, float]:
    dmg_to_1 = sum(max(0, prev_towers[1][k] - self.towers[1][k]) for k in prev_towers[1].keys())
    dmg_to_0 = sum(max(0, prev_towers[0][k] - self.towers[0][k]) for k in prev_towers[0].keys())
    base0 = float(dmg_to_1 - dmg_to_0)
    base1 = float(dmg_to_0 - dmg_to_1)
    # scale by small weight
    return self.cfg.damage_weight * base0, self.cfg.damage_weight * base1

  def step(self, a0_card: int, a0_lane: int, a0_place: int, a1_card: int, a1_lane: int, a1_place: int):
    if self.is_done():
      raise RuntimeError("Game over")
    prev_towers = {0: dict(self.towers[0]), 1: dict(self.towers[1])}

    for team in [0, 1]:
      ps = self.players[team]
      ps.resource = min(self.cfg.max_resource, ps.resource + self.cfg.resource_per_turn)

    for team, card_action, lane_action, place_action in [(0, a0_card, a0_lane, a0_place),
                                                         (1, a1_card, a1_lane, a1_place)]:
      ps = self.players[team]
      legal_cards = self.legal_card_actions(team)
      if card_action not in legal_cards:
        card_action = 2
      if card_action in [0, 1] and card_action < len(ps.deck_queue):
        idx = ps.deck_queue[card_action]
        card = ps.deck[idx]
        if ps.resource >= card.cost:
          self._spawn_unit(team, card, lane_action, place_action)
          ps.resource -= card.cost
          self._draw_replacement(team, card_action)

    self._advance_units()
    self._apply_lead_damage()
    self.units = [u for u in self.units if u.is_alive()]

    dmg_units, dmg_towers, attacking_king = self._collect_targets()
    self._apply_king_counter_damage(attacking_king)
    self._apply_damage(dmg_units, dmg_towers)
    self.turn += 1
    rew0, rew1 = self._compute_reward(prev_towers)
    # terminal bonus
    done_flag = self.is_done()
    if done_flag:
      winner = self.compute_tiebreak_winner()
      if winner == 0:
        rew0 += self.cfg.win_reward
        rew1 -= self.cfg.win_reward
      elif winner == 1:
        rew0 -= self.cfg.win_reward
        rew1 += self.cfg.win_reward
    obs0, obs1 = self.get_observation(0), self.get_observation(1)
    return (obs0, obs1), (rew0 * self.cfg.reward_multiplier, rew1 * self.cfg.reward_multiplier), done_flag, {}

  def is_done(self) -> bool:
    if self.turn >= self.cfg.max_turns:
      return True
    for team in [0, 1]:
      if self.towers[team]["king"] <= 0:
        return True
    return False

  def get_observation(self, team: Team) -> Dict[str, np.ndarray]:
    grid = np.zeros((self.cfg.board_h, self.cfg.board_w), dtype=np.int8)
    for u in self.units:
      mark = 1 if u.team == team else -1
      grid[u.row, u.col] = mark
    grid[self.cfg.river_row, :] = 2
    my_towers = np.array([self.towers[team]["king"], self.towers[team]["left"], self.towers[team]["right"]],
                         dtype=np.int16)
    opp = 1 - team
    opp_towers = np.array([self.towers[opp]["king"], self.towers[opp]["left"], self.towers[opp]["right"]],
                          dtype=np.int16)
    ps = self.players[team]
    hand = np.array(ps.deck_queue[:2], dtype=np.int8)
    resource = np.array(ps.resource, dtype=np.int16)
    return {"grid": grid, "my_towers": my_towers, "opp_towers": opp_towers, "hand": hand, "resource": resource}

  def render_ascii(self, perspective: Team = 0) -> str:
    grid_chars = [["." for _ in range(self.cfg.board_w)] for _ in range(self.cfg.board_h)]
    for c in range(self.cfg.board_w):
      grid_chars[self.cfg.river_row][c] = "~"
    for team, towers in self.tower_positions.items():
      for name, (r, c) in towers.items():
        if self.towers[team][name] <= 0:
          continue
        ch = {"king": "k", "left": "l", "right": "r"}[name]
        if team == perspective:
          ch = ch.upper()
        grid_chars[r][c] = ch
    for u in self.units:
      ch = u.unit_type.symbol
      ch = ch.upper() if u.team == perspective else ch.lower()
      r, c = u.row, u.col
      if grid_chars[r][c] != "." and grid_chars[r][c] != "~":
        grid_chars[r][c] = "X"
      else:
        grid_chars[r][c] = ch

    lines: List[str] = ["".join(row) for row in grid_chars]

    obs = self.get_observation(perspective)
    my_t = obs["my_towers"].tolist()
    opp_t = obs["opp_towers"].tolist()
    lines.append(f"My Towers [K,L,R]: {my_t}")
    lines.append(f"Opp Towers [K,L,R]: {opp_t}")
    ps = self.players[perspective]
    hand_names = []
    for slot in [0, 1]:
      if slot < len(ps.deck_queue):
        idx = ps.deck_queue[slot]
        hand_names.append(f"{ps.deck[idx].name}({ps.deck[idx].cost})")
    lines.append(f"Hand: {hand_names}")
    lines.append(f"Resource: {int(obs['resource'])}/{self.cfg.max_resource}")
    lines.append("Lane actions: 0=left, 1=right | Place: 0=near, 1=far")

    units_sorted = sorted(self.units, key=lambda u: (u.team != perspective, u.row, u.col))
    unit_summ = [
      f"{u.unit_type.name[0]}{'+' if u.team==perspective else '-'}@({u.row},{u.col}) hp={u.hp}" for u in units_sorted
    ]
    if unit_summ:
      lines.append("Units: " + ", ".join(unit_summ))

    return "\n".join(lines)

  def compute_tiebreak_winner(self) -> int:
    """Return 0 if team 0 wins, 1 if team 1 wins, -1 for draw.
    Rules:
    1) More surviving towers (hp>0) wins.
    2) If equal, compare minimum positive tower HP; higher minimum wins.
    3) If equal, draw.
    """
    t0 = [hp for hp in self.towers[0].values() if hp > 0]
    t1 = [hp for hp in self.towers[1].values() if hp > 0]
    if len(t0) != len(t1):
      return 0 if len(t0) > len(t1) else 1
    min0 = min(t0) if t0 else 0
    min1 = min(t1) if t1 else 0
    if min0 == min1:
      return -1
    return 0 if min0 > min1 else 1

  def self_play_episode(self,
                        policy0,
                        policy1,
                        render: bool = False,
                        return_winner: bool = False) -> Tuple[float, float, int]:
    self.reset()
    R0, R1 = 0.0, 0.0
    while not self.is_done():
      obs0, obs1 = self.get_observation(0), self.get_observation(1)
      a0_card = policy0.select_card_action(obs0, self.legal_card_actions(0))
      a0_lane = policy0.select_lane_action(obs0, self.legal_lane_actions(0))
      a0_place = policy0.select_place_action(obs0, self.legal_place_actions(0))
      a1_card = policy1.select_card_action(obs1, self.legal_card_actions(1))
      a1_lane = policy1.select_lane_action(obs1, self.legal_lane_actions(1))
      a1_place = policy1.select_place_action(obs1, self.legal_place_actions(1))
      (_, _), (r0, r1), done, _ = self.step(a0_card, a0_lane, a0_place, a1_card, a1_lane, a1_place)
      R0 += r0
      R1 += r1
      if render:
        print(self.render_ascii(0))
        print("-" * 32)
    if return_winner:
      winner = self.compute_tiebreak_winner()
      return R0, R1, self.turn, winner  # type: ignore[return-value]
    return R0, R1, self.turn  # type: ignore[return-value]

  def human_vs_agent(self, agent_policy, human_team: Team = 0):
    self.reset()
    while not self.is_done():
      persp = human_team
      print(self.render_ascii(persp))
      ps = self.players[persp]
      print("Choose card: 0=front, 1=pre-front, 2=skip | lane: 0=left,1=right | place: 0=near,1=far")
      try:
        card_choice = int(input("Card> ").strip())
      except Exception:
        card_choice = 2
      legal_cards = self.legal_card_actions(persp)
      card_choice = card_choice if card_choice in legal_cards else 2
      lane_choice = 0
      place_choice = 0
      if card_choice in [0, 1]:
        try:
          lane_choice = int(input("Lane> ").strip())
        except Exception:
          lane_choice = 0
        lane_choice = lane_choice if lane_choice in [0, 1] else 0
        try:
          place_choice = int(input("Place> ").strip())
        except Exception:
          place_choice = 0
        place_choice = place_choice if place_choice in [0, 1] else 0
      opp_team = 1 - human_team
      opp_obs = self.get_observation(opp_team)
      opp_card = agent_policy.select_card_action(opp_obs, self.legal_card_actions(opp_team))
      opp_lane = agent_policy.select_lane_action(opp_obs, self.legal_lane_actions(opp_team))
      opp_place = agent_policy.select_place_action(opp_obs, self.legal_place_actions(opp_team))
      if human_team == 0:
        self.step(card_choice, lane_choice, place_choice, opp_card, opp_lane, opp_place)
      else:
        self.step(opp_card, opp_lane, opp_place, card_choice, lane_choice, place_choice)
    print("Game over")
    winner = self.compute_tiebreak_winner()
    if winner == -1:
      print("Result: Draw")
    else:
      print("Result: You win" if winner == human_team else "Result: Bot wins")
    print(self.render_ascii(human_team))


class RandomPolicy:

  def select_card_action(self, obs, legal_card_actions: List[int]) -> int:
    return random.choice(legal_card_actions)

  def select_lane_action(self, obs, legal_lane_actions: List[int]) -> int:
    return random.choice(legal_lane_actions)

  def select_place_action(self, obs, legal_place_actions: List[int]) -> int:
    return random.choice(legal_place_actions)

  def select_action(self, obs, legal_actions: List[int]):
    return self.select_card_action(obs, legal_actions), self.select_lane_action(obs, [0, 1])


if __name__ == "__main__":
  engine = CREngine()
  engine.human_vs_agent(RandomPolicy(), human_team=1)
