import numpy as np
import gymnasium as gym
from gym import spaces
from pettingzoo import ParallelEnv

class HierarchicalJammingEnv(ParallelEnv):
    def __init__(self, hop_pts, jamming_bandwidths, diversity, num_radars, num_jammers, max_steps, max_hop_length):
        super().__init__()

        self.hop_pts = hop_pts[:num_radars]
        self.max_hop_length = max_hop_length

        self.low = [hop[0] for hop in self.hop_pts]
        self.interval = [hop[1] - hop[0] for hop in self.hop_pts]
        self.n_frequencies = [len(hop) for hop in self.hop_pts]
        self.max_steps = max_steps

        self.agents = [f"jammer_{i}" for i in range(num_jammers)]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = {agent: i for i, agent in enumerate(self.agents)}

        self.jamming_bandwidths = jamming_bandwidths
        self.n_bandwidths = len(self.jamming_bandwidths)
        self.diversity = diversity

        self.action_spaces = {agent: spaces.Dict({
            "radar": spaces.Discrete(len(self.hop_pts)),
            "frequency": spaces.Discrete(self.n_frequencies[0]),
            "bandwidth": spaces.Discrete(self.n_bandwidths),
        }) for agent in self.agents}

        self.observation_spaces = {agent: spaces.Dict({
            "selected_radar": spaces.Discrete(len(self.hop_pts)),
            "selected_radar_frequencies": spaces.MultiBinary(self.n_frequencies[0]),
            "all_agents_actions": spaces.MultiDiscrete([len(self.hop_pts), self.n_frequencies[0], self.n_bandwidths] * len(self.agents)),
        }) for agent in self.agents}

        #init a random array of threat levels such that the sum of the threat levels is 10
        self.threat_levels = np.random.randint(2, 5, len(self.hop_pts))
        self.threat_levels = (self.threat_levels / np.sum(self.threat_levels)) * 10

        self.hopping_patterns = None
        self.current_steps = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.rewards = {agent: 0 for agent in self.agents}
        self.radar_frequencies = None
        self.last_actions = {agent: None for agent in self.agents}

        #visualization attributes
        self.num_jammed_freqs = np.zeros((len(self.hop_pts), len(self.hop_pts[0])))

    def get_environment_state(self):
        state = []
        for radar in self.hop_pts:
            radar_state = np.zeros(len(radar))
            for freq in self.radar_frequencies[self.hop_pts.index(radar)]:
                freq_index = np.where(np.array(radar) == freq)[0][0]
                radar_state[freq_index] = 1
            state.append(radar_state.tolist())
        return state
    
    def store_previous_actions(self):
        """
        Copies the current actions to the previous_actions attribute.
        This should be called at the end of the step function so that the next observation
        reflects the effects of the actions taken in the last step.
        """
        self.previous_actions = {agent: action for agent, action in self.last_actions.items()}

    def observe(self, agent):
        """
        The observation includes:
        selected_radar_frequencies: A MultiDiscrete array (as a numpy int array) where each frequency is:
             1  if the frequency was transmitted and jammed,
            -2  if transmitted but not jammed,
            -1  if not transmitted but jammed,
             0  if neither transmitted nor jammed.
        all_agents_actions: A flat numpy array concatenating the previous (radar, frequency, bandwidth)
            actions for all agents.
        
        For the first step (i.e. when no previous actions exist), a default observation is provided.
        """
        # Provide default observation if no previous actions exist for this agent.
        if self.previous_actions[agent] is None:
            default_transmitted = np.zeros(self.n_frequencies[0], dtype=int)
            default_all_actions = np.array([-1, -1, -1] * len(self.agents))
            default_selected_radar = -1
            return {
                "selected_radar": default_selected_radar,
                "selected_radar_frequencies": default_transmitted,
                "all_agents_actions": default_all_actions
            }
        
        # Retrieve the radar selected by the agent in the previous step.
        radar_choice = self.previous_actions[agent].get("radar", 0)
        radar_freqs = self.hop_pts[radar_choice]
        n_freq = len(radar_freqs)
        
        # Assume transmitted frequencies for the radar come from self.radar_frequencies.
        # (This set should have been updated in reset/step to reflect the frequencies transmitted by this radar.)
        transmitted_set = set(self.radar_frequencies[radar_choice]) if (self.radar_frequencies and len(self.radar_frequencies) > radar_choice) else set()
        
        # Build a list of jamming ranges based on previous actions of all agents.
        jam_ranges = []
        for a in self.agents:
            action = self.previous_actions[a]
            if action is not None:
                a_radar = action.get("radar", 0)
                a_freq_idx = action.get("frequency", 0)
                # Validate indices before proceeding.
                if a_radar < len(self.hop_pts) and a_freq_idx < len(self.hop_pts[a_radar]):
                    jam_freq = self.hop_pts[a_radar][a_freq_idx]
                    bw_index = action.get("bandwidth", 0)
                    if bw_index < len(self.jamming_bandwidths):
                        bw = self.jamming_bandwidths[bw_index]
                        lower_bound = jam_freq - bw / 2
                        upper_bound = jam_freq + bw / 2
                        jam_ranges.append((lower_bound, upper_bound))
        
        # Precompute a mapping from frequency value to its index in radar_freqs.
        freq_to_index = {freq: idx for idx, freq in enumerate(radar_freqs)}
        
        # Build the observation array based on whether each frequency was transmitted and/or jammed.
        obs_array = np.zeros(n_freq, dtype=int)
        for freq in radar_freqs:
            transmitted = freq in transmitted_set
            jammed = any(lower_bound <= freq <= upper_bound for (lower_bound, upper_bound) in jam_ranges)
            
            if transmitted and jammed:
                value = 1
                self.num_jammed_freqs[radar_choice][freq_to_index[freq]] = 1

            elif transmitted and not jammed:
                value = -2
            elif (not transmitted) and jammed:
                value = -1
            else:
                value = 0
            
            idx = freq_to_index[freq]
            obs_array[idx] = value

        all_actions = []
        for a in self.agents:
            action = self.previous_actions[a] if self.previous_actions[a] is not None else {"radar": -1, "frequency": -1, "bandwidth": -1}
            all_actions.extend([action["radar"], action["frequency"], action["bandwidth"]])
        
        return {
            "selected_radar": radar_choice,
            "selected_radar_frequencies": obs_array,
            "all_agents_actions": np.array(all_actions)
        }


    def generate_hopping_pattern(self):
        return [np.random.choice(hop, self.max_hop_length, replace=False) for hop in self.hop_pts]

    def reset(self):
        self.hopping_patterns = self.generate_hopping_pattern()
        self.radar_frequencies = [pattern[:self.diversity] for pattern in self.hopping_patterns]
        self.current_steps = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.rewards = {agent: 0 for agent in self.agents}
        self.last_actions = {agent: None for agent in self.agents}
        self.previous_actions = {agent: None for agent in self.agents}
        self.state = self.get_environment_state()
        # self.threat_levels = np.random.randint(1, 5, len(self.hop_pts))
        # self.threat_levels = (self.threat_levels / np.sum(self.threat_levels)) * 10
        self.num_jammed_freqs = np.zeros((len(self.hop_pts), len(self.hop_pts[0])))
        return {agent: self.observe(agent) for agent in self.agents}, {agent: False for agent in self.agents}
    
    def step(self, actions):
        # Update current actions
        for agent, action in actions.items():
            self.last_actions[agent] = action

        for agent in self.agents:
            if self.terminations[agent]:
                self._was_dead_step(actions[agent])
                return

        self.state = self.get_environment_state()

        radar_actions = [action["radar"] for action in actions.values()]
        freq_actions = [action["frequency"] for action in actions.values()]
        bw_actions = [action["bandwidth"] for action in actions.values()]
        rewards = [0 for _ in self.agents]

        jammed_freqs = [self.hop_pts[radar][freq] for radar, freq in zip(radar_actions, freq_actions)]
        upper_bounds = [jam_freq + self.jamming_bandwidths[bw] / 2 for jam_freq, bw in zip(jammed_freqs, bw_actions)]
        lower_bounds = [jam_freq - self.jamming_bandwidths[bw] / 2 for jam_freq, bw in zip(jammed_freqs, bw_actions)]

        for i, (low, high) in enumerate(zip(lower_bounds, upper_bounds)):
            jammed = False
            for radar_freq_list in self.radar_frequencies:
                for radar_freq in radar_freq_list:
                    if low <= radar_freq <= high:
                        rewards[i] += 10 / self.max_steps * self.threat_levels[radar_actions[i]]
                        jammed = True
            if not jammed:
                rewards[i] -= 30 / self.max_steps

        for i in range(len(bw_actions)):
            if bw_actions[i] > self.interval[radar_actions[i]] * self.diversity:
                rewards[i] -= 10 / self.max_steps

        for i in range(len(radar_actions)):
            for j in range(i + 1, len(radar_actions)):
                if radar_actions[i] == radar_actions[j] and freq_actions[i] == freq_actions[j]:
                    rewards[j] -= 15 / self.max_steps

        self.rewards = {agent: rewards[i] for i, agent in enumerate(self.agents)}

        for agent in self.agents:
            self.current_steps[agent] += 1

        for agent in self.agents:
            if self.current_steps[agent] >= self.max_steps:
                self.terminations[agent] = True

        self.radar_frequencies = [
            [pattern[(self.current_steps[agent] + i) % self.max_hop_length] for i in range(self.diversity)]
            for pattern in self.hopping_patterns
        ]

        self.store_previous_actions()
        self.last_actions = {agent: actions[agent] for agent in self.agents}

        observations = {agent: self.observe(agent) for agent in self.agents}
        rewards = {agent: rewards[i] for i, agent in enumerate(self.agents)}
        terminations = {agent: self.terminations[agent] for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        info = {agent: {} for agent in self.agents}

        return observations, rewards, terminations, truncations, self.num_jammed_freqs, info

    def render(self):
        for agent in self.agents:
            print(f"Agent {agent}, Radar Frequencies: {self.radar_frequencies}")

    def close(self):
        pass
