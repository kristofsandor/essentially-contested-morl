import gymnasium as gym
import numpy as np

class WindowObservationWrapper(gym.ObservationWrapper):

    def __init__(self, env: gym.Env, window_size = 1):
        super().__init__(env)

        self.window_size = window_size

        self.observation_space = gym.spaces.MultiDiscrete([window_size*window_size])
    
    def observation(self, obs) -> int:
        """Cuts out a padded window of size (window_size, window_size) around the player."""
        # obs is a 16x16x6 array
        ws = self.window_size

        # Find player position in the first channel
        player_flat_idx = np.argmax(obs[:, :, 0])
        player_x, player_y = np.unravel_index(player_flat_idx, obs[:, :, 0].shape)

        # Pad the observation so we can always take a fixed-size window,
        # even when the player is close to the borders.
        pad = ws // 2
        padded_obs = np.pad(
            obs,
            pad_width=((pad, pad), (pad, pad), (0, 0)),
            mode="constant",
        )

        # Player position in padded coordinates
        px, py = player_x + pad, player_y + pad

        # Extract a fixed ws x ws window centered on the player
        x_start = px - ws
        x_end = px + ws
        y_start = py - ws
        y_end = py + ws

        window = padded_obs[x_start:x_end, y_start:y_end, :]
        flat_window = window.flatten()

        return flat_window
            