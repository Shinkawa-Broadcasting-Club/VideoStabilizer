from __future__ import annotations

import unittest

from video_stabilizer.config import Config
from video_stabilizer.gui.settings_store import config_from_ui_settings


class TestSettingsStore(unittest.TestCase):
    def test_config_from_ui_settings_ignores_invalid_numeric(self) -> None:
        base = Config()
        ui = {
            "ema_alpha": "abc",
            "ratio_clip_low": "x",
            "ratio_clip_high": None,
            "target_sampling_sec": "NaN",
            "output_dir": "",
        }
        cfg = config_from_ui_settings(ui, base)
        self.assertEqual(cfg.ema_alpha, base.ema_alpha)
        self.assertEqual(cfg.ratio_clip_low, base.ratio_clip_low)
        self.assertEqual(cfg.ratio_clip_high, base.ratio_clip_high)
        self.assertEqual(cfg.target_sampling_sec, base.target_sampling_sec)


if __name__ == "__main__":
    unittest.main()

