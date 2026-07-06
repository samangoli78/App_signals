"""Self-contained shell: collapsible ribbon + dock grid panel hosts.



Run the standalone demo::



    python -m App.shell

"""



from .app_frame import (

    DEFAULT_PANELS,

    PanelSpec,

    ShellFrame,

    build_dock_layout,

    build_shell_frame,

)

from .dock_grid import DockGridLayout

from .ribbon import (

    RibbonSection,

    RibbonShell,

    build_ribbon_shell,

    collapsible_section,

    make_scrollable_column,

    ribbon_button,

    ribbon_checkbox,

    ribbon_label,

)



__all__ = [

    "DEFAULT_PANELS",

    "DockGridLayout",

    "PanelSpec",

    "RibbonSection",

    "RibbonShell",

    "ShellFrame",

    "build_dock_layout",

    "build_ribbon_shell",

    "build_shell_frame",

    "collapsible_section",

    "make_scrollable_column",

    "ribbon_button",

    "ribbon_checkbox",

    "ribbon_label",

]


