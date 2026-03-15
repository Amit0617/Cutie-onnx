__all__ = ["MainControllerOnnxNumpy"]


def __getattr__(name):
    if name == "MainControllerOnnxNumpy":
        from .main_controller import MainControllerOnnxNumpy

        return MainControllerOnnxNumpy
    raise AttributeError(name)
