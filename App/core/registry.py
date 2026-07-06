"""Track open main-app instances (multi-window / debugging)."""

_instances: list = []


def register_app(instance) -> None:
    _instances.append(instance)


def open_apps():
    return list(_instances)
