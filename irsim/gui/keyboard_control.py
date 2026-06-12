from typing import Any, Optional, Dict, List
import numpy as np
import matplotlib.pyplot as plt

# Optional pynput backend
keyboard = None
_PYNPUT_AVAILABLE = False

try:
    from pynput import keyboard as _pynput_keyboard
    if getattr(_pynput_keyboard, "Listener", None) is not None:
        keyboard = _pynput_keyboard
        _PYNPUT_AVAILABLE = True
except Exception:
    _PYNPUT_AVAILABLE = False


class KeyboardControl:
    """
    Clean IR-Sim keyboard controller with configurable key mapping.

    Key improvement:
        - ALL key bindings are defined in self.keymap
        - No duplicated logic between matplotlib and pynput
        - Easy to extend for teaching / Olympiad training
    """

    # ------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------
    def __init__(self, env_ref: Any | None = None, **kwargs: Any) -> None:
        self.env_ref = env_ref

        # -----------------------------
        # Control parameters
        # -----------------------------
        self.key_lv_max = kwargs.get("key_lv_max", 2.0)
        self.key_ang_max = kwargs.get("key_ang_max", 0.5)

        self.key_lv = 0.0
        self.key_ang = 0.0
        self.key_rot = 0.0
        self.key_id = kwargs.get("key_id", 0)

        # -----------------------------
        # Backend selection
        # -----------------------------
        self.backend = kwargs.get("backend", "pynput").lower()
        if self.backend not in ["pynput", "mpl"]:
            self.backend = "mpl"

        if self.backend == "pynput" and not _PYNPUT_AVAILABLE:
            self.backend = "mpl"

        # -----------------------------
        # KEY MAPPING (MAIN FEATURE)
        # -----------------------------
        self.keymap = kwargs.get("keymap", self._default_keymap())

        # Reverse map for fast lookup
        self.key_to_action = self._build_reverse_map(self.keymap)

        # -----------------------------
        # State
        # -----------------------------
        self.alt_flag = False
        self.key_vel = np.zeros((3, 1))

        # -----------------------------
        # Matplotlib integration
        # -----------------------------
        try:
            fig = plt.gcf()
            fig.canvas.mpl_connect("key_press_event", self._on_mpl_press)
            fig.canvas.mpl_connect("key_release_event", self._on_mpl_release)
        except Exception:
            pass

        # -----------------------------
        # Pynput listener
        # -----------------------------
        if self.backend == "pynput" and _PYNPUT_AVAILABLE:
            self.listener = keyboard.Listener(
                on_press=self._on_pynput_press,
                on_release=self._on_pynput_release,
            )
            self.listener.start()

        # -----------------------------
        # Info print
        # -----------------------------
        if self._world_param.control_mode == "keyboard":
            self._print_controls()

    # ------------------------------------------------------------
    # Default keymap
    # ------------------------------------------------------------
    def _default_keymap(self) -> Dict[str, List[str]]:
        return {
            "forward": ["w", "up"],
            "backward": ["s", "down"],
            "left": ["a", "left"],
            "right": ["d", "right"],
            "rot_left": ["q"],
            "rot_right": ["e"],
            "stop": ["space"],
        }

    def _build_reverse_map(self, keymap: Dict[str, List[str]]) -> Dict[str, str]:
        reverse = {}
        for action, keys in keymap.items():
            for k in keys:
                reverse[k] = action
        return reverse

    # ------------------------------------------------------------
    # Action application (single source of truth)
    # ------------------------------------------------------------
    def _apply_action(self, action: str, pressed: bool) -> None:
        if action == "forward":
            self.key_lv = self.key_lv_max if pressed else 0.0

        elif action == "backward":
            self.key_lv = -self.key_lv_max if pressed else 0.0

        elif action == "left":
            self.key_ang = self.key_ang_max if pressed else 0.0

        elif action == "right":
            self.key_ang = -self.key_ang_max if pressed else 0.0

        elif action == "rot_left":
            self.key_rot = self.key_ang_max if pressed else 0.0

        elif action == "rot_right":
            self.key_rot = -self.key_ang_max if pressed else 0.0

        self._update_velocity()

    def _update_velocity(self):
        self.key_vel = np.array([[self.key_lv], [self.key_ang], [self.key_rot]])

    def _resolve_key(self, key: str):
        return self.key_to_action.get(key)

    # ------------------------------------------------------------
    # pynput backend
    # ------------------------------------------------------------
    def _on_pynput_press(self, key: Any) -> None:
        if self._world_param.control_mode != "keyboard":
            return

        try:
            k = key.char.lower()
        except Exception:
            return

        action = self._resolve_key(k)
        if action:
            self._apply_action(action, True)

    def _on_pynput_release(self, key: Any) -> None:
        if self._world_param.control_mode != "keyboard":
            return

        try:
            k = key.char.lower()
        except Exception:
            return

        action = self._resolve_key(k)
        if action:
            self._apply_action(action, False)

        # ---- system keys ----
        self._handle_system_keys_pynput(key)

    def _handle_system_keys_pynput(self, key: Any):
        try:
            if keyboard is not None and key == keyboard.Key.space:
                self._toggle_pause()

            if keyboard is not None and key == keyboard.Key.esc:
                self.env_ref.quit_flag = True

        except Exception:
            pass

    # ------------------------------------------------------------
    # Matplotlib backend
    # ------------------------------------------------------------
    def _on_mpl_press(self, event: Any) -> None:
        if self._world_param.control_mode != "keyboard":
            return

        key = (event.key or "").lower()
        action = self._resolve_key(key)

        if action:
            self._apply_action(action, True)

    def _on_mpl_release(self, event: Any) -> None:
        if self._world_param.control_mode != "keyboard":
            return

        key = (event.key or "").lower()
        action = self._resolve_key(key)

        if action:
            self._apply_action(action, False)

        if key == "space":
            self._toggle_pause()

        if key == "escape":
            self.env_ref.quit_flag = True

    # ------------------------------------------------------------
    # System actions
    # ------------------------------------------------------------
    def _toggle_pause(self):
        if "Pause" not in self.env_ref.status:
            self.env_ref.pause()
        else:
            self.env_ref.resume()

    # ------------------------------------------------------------
    # UI
    # ------------------------------------------------------------
    def _print_controls(self):
        print("\n=== IR-Sim Keyboard Control ===")
        for action, keys in self.keymap.items():
            print(f"{action:12s}: {', '.join(keys)}")
        print("===============================\n")

    # ------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------
    @property
    def _world_param(self):
        if self.env_ref is not None:
            return self.env_ref._world_param
        from irsim.config import world_param
        return world_param
