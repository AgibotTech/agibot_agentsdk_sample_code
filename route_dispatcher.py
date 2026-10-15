from __future__ import annotations

import importlib
import logging
import re
from pathlib import Path

from app_config import (
    CONTROL_ROUTE_NAMES,
    DEFAULT_CONFIG_PATH,
    ReplyRouteConfig,
    load_reply_route_configs,
)
from base_handler import BaseHandler, HandlerContext, ReplyHandler
from scope_router import ScopeRoute, route_name


logger = logging.getLogger(__name__)

HANDLER_REFERENCE_PATTERN = re.compile(
    r"^handlers\.(?P<module>[a-z][a-z0-9_]*_handler):"
    r"(?P<class>[A-Za-z_][A-Za-z0-9_]*)$"
)


class RouteDispatcher:
    """Dispatch configured reply routes to their dedicated handlers."""

    def __init__(self, handlers: dict[str, ReplyHandler]) -> None:
        invalid_routes = set(handlers) & CONTROL_ROUTE_NAMES
        if invalid_routes:
            names = ", ".join(sorted(invalid_routes))
            raise ValueError(f"Control routes cannot use reply handlers: {names}")
        if not handlers:
            raise ValueError("At least one reply handler must be registered")
        self._handlers = dict(handlers)

    def dispatch(
        self,
        route: ScopeRoute | str,
        context: HandlerContext,
    ) -> str:
        name = route_name(route)
        handler = self._handlers.get(name)
        if handler is None:
            raise ValueError(f"No reply handler registered for route: {name}")
        return handler.handle(context)

    def close(self) -> None:
        closed_handler_ids: set[int] = set()
        for name, handler in self._handlers.items():
            handler_id = id(handler)
            if handler_id in closed_handler_ids:
                continue
            closed_handler_ids.add(handler_id)
            try:
                handler.close()
            except Exception:  # noqa: BLE001 - close all remaining handlers.
                logger.exception("reply_handler_close_failed route=%s", name)


def build_route_dispatcher(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> RouteDispatcher:
    """Load and instantiate every reply handler registered in config.json."""

    handlers: dict[str, ReplyHandler] = {}
    for config in load_reply_route_configs(config_path):
        handler_class = _load_handler_class(config)
        handlers[config.name] = handler_class(config)
    return RouteDispatcher(handlers)


def _load_handler_class(config: ReplyRouteConfig) -> type[BaseHandler]:
    match = HANDLER_REFERENCE_PATTERN.fullmatch(config.handler)
    if match is None:
        raise ValueError(
            "reply_routes."
            f"{config.name}.handler must use handlers.module_handler:ClassName format"
        )

    handler_module = match.group("module")
    module_name = f"handlers.{handler_module}"
    class_name = match.group("class")
    project_dir = Path(__file__).resolve().parent
    handlers_dir = project_dir / "handlers"
    module_path = handlers_dir / f"{handler_module}.py"
    if not module_path.is_file() or module_path.resolve().parent != handlers_dir:
        raise ValueError(
            "Reply handler module must be a handlers/*_handler.py file: "
            f"{module_name}"
        )

    module = importlib.import_module(module_name)
    loaded_path = getattr(module, "__file__", None)
    if loaded_path is None or Path(loaded_path).resolve() != module_path.resolve():
        raise ValueError(f"Reply handler module is outside the project: {module_name}")
    handler_class = getattr(module, class_name, None)
    if not isinstance(handler_class, type) or not issubclass(handler_class, BaseHandler):
        raise ValueError(f"Reply handler must inherit BaseHandler: {config.handler}")
    return handler_class
