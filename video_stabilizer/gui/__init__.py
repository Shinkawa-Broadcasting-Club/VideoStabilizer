# GUI package


def launch_gui() -> None:
    from video_stabilizer.gui.app_window import launch_gui as _impl

    _impl()


__all__ = ["launch_gui"]
