from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseController(ABC):
    def __init__(self, app) -> None:
        self.app = app

    @abstractmethod
    def p_increase(self, event=None): ...

    @abstractmethod
    def p_decrease(self, event=None): ...

    @abstractmethod
    def s_increase(self, event=None): ...

    @abstractmethod
    def s_decrease(self, event=None): ...


class BasePlotPresenter(ABC):
    def __init__(self, app) -> None:
        self.app = app

    @abstractmethod
    def on_right_click(self, event): ...

    @abstractmethod
    def on_right_follow_mouse(self, event): ...

    @abstractmethod
    def on_right_release(self, event): ...

    @abstractmethod
    def on_scroll(self, event): ...

    @abstractmethod
    def plot_main(self, ax, x, y2, arg="top", reff=None): ...

    @abstractmethod
    def plot(self): ...

    @abstractmethod
    def create_legend(self, leg, canvas, addition=None): ...

    @abstractmethod
    def energy(self, ax, x, y, legends=None): ...


class BaseSessionStore(ABC):
    @abstractmethod
    def save_delta(self, path: str, delta: list[Any]) -> None: ...

    @abstractmethod
    def load_delta(self, path: str) -> list[Any]: ...


class BaseCartoParser(ABC):
    @abstractmethod
    def parse_mesh_file(self): ...

    @abstractmethod
    def mesh_build(self): ...

    @abstractmethod
    def pars_mesh_file_with_electrode(self): ...

    @abstractmethod
    def save_mesh_toVTK(self, fname="output.vtp", abs_path=None): ...


class BaseCartoLoader(ABC):
    @abstractmethod
    def on_init(self): ...

    @abstractmethod
    def Signals(self, triple=False): ...


class BaseTableAppGlue(ABC):
    def __init__(self, app) -> None:
        self.app = app

    @abstractmethod
    def table_select_ctx(self, ctx): ...

    @abstractmethod
    def table_move_ctx(self, ctx): ...

    @abstractmethod
    def table_commit_ctx(self, ctx): ...


class BaseMeshAppGlue(ABC):
    """Bridge between the 3D mesh viewer and the rest of the app (table + plots)."""

    def __init__(self, app) -> None:
        self.app = app
        self.viewer = None

    @abstractmethod
    def attach(self, panel) -> None: ...

    @abstractmethod
    def sync_from_app(self) -> None: ...

    @abstractmethod
    def on_pick(self, kind: str, payload, info: dict) -> None: ...
