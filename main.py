from __future__ import annotations

import json
import logging
import random
import time

from linksoul_agentsdk import (
    AgentAuthCallback,
    AgentParam,
    AgentSdk,
    IdGenerator,
)
from linksoul_agentsdk.passive import AsrVideo2VlmCallback, AsrVideo2VlmResponse

from app_config import load_agent_config, load_reply_route_configs
from base_handler import HandlerContext
from conversation_manager import ConversationManager
from route_dispatcher import build_route_dispatcher
from skill_handler import SkillHandler
from scope_router import ScopeRoute, ScopeRouter, route_name


logger = logging.getLogger(__name__)

_STREAMING_DELTA_EVENT_TYPES = (
    "agentsdk.llm_response.item.delta",
    "agentsdk.vlm_response.item.delta",
)


class _SdkStreamingDeltaLogFilter(logging.Filter):
    """Hide successful per-chunk SDK logs while preserving failures and lifecycle logs."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno > logging.INFO:
            return True
        message = record.getMessage()
        return not any(event_type in message for event_type in _STREAMING_DELTA_EVENT_TYPES)


def log_conversation(
    *,
    agent_id: str,
    event_id: str,
    user_text: str,
    intent: str,
    assistant_text: str,
) -> None:
    logger.info(
        "conversation_user_input agent_id=%s event_id=%s user_text=%s",
        agent_id,
        event_id,
        json.dumps(user_text, ensure_ascii=False),
    )
    logger.info(
        "conversation_intent agent_id=%s event_id=%s intent=%s",
        agent_id,
        event_id,
        intent,
    )
    logger.info(
        "conversation_robot_response agent_id=%s event_id=%s assistant_text=%s",
        agent_id,
        event_id,
        json.dumps(assistant_text, ensure_ascii=False),
    )


# Authentication callback.
class MyAuth(AgentAuthCallback):
    def on_auth_state(self, app_id: str, code: int, msg: str) -> None:
        if code == 0:
            logger.info("agent_auth_succeeded app_id=%s", app_id)
        else:
            logger.error("agent_auth_failed app_id=%s code=%s message=%s", app_id, code, msg)


def load_llm_waiting_replies(config_path: str = "config.json") -> dict[str, tuple[str, ...]]:
    return {
        route.name: route.waiting_replies
        for route in load_reply_route_configs(config_path)
    }


def load_llm_timeout_replies(config_path: str = "config.json") -> dict[str, str]:
    return {
        route.name: route.timeout_reply_text
        for route in load_reply_route_configs(config_path)
    }


def build_llm_waiting_reply(replies: tuple[str, ...]) -> str | None:
    if not replies:
        return None
    return random.choice(replies)


def get_llm_waiting_replies(
    replies_by_route: dict[str, tuple[str, ...]],
    route: str,
) -> tuple[str, ...]:
    return replies_by_route.get(route, ())


class _Cb(AsrVideo2VlmCallback):
    def __init__(self, agent_sdk: AgentSdk) -> None:
        super().__init__(agent_sdk)
        # Runtime behavior is loaded from the project and locale configuration.
        self.scope_router = ScopeRouter.from_config_file(
            config_path="config.json",
        )
        self.route_dispatcher = build_route_dispatcher(config_path="config.json")
        self.skill_handler = SkillHandler(config_path="config.json")
        self.llm_waiting_replies = load_llm_waiting_replies("config.json")
        self.llm_timeout_replies = load_llm_timeout_replies("config.json")
        self.conversation = ConversationManager.from_config_file("config.json")
        self.tts_state: str | None = None

    def close(self) -> None:
        for component in (
            self.scope_router,
            self.route_dispatcher,
            self.skill_handler,
        ):
            try:
                component.close()
            except Exception:  # noqa: BLE001 - continue closing remaining clients.
                logger.exception("llm_client_close_failed")
        self.conversation.clear()

    def on_request(
        self, agent_id, event_id, flag, text, buf, param,
        response: AsrVideo2VlmResponse,
    ):
        """flag: 3 ASR text, 4 H264 video frame, 5 image frame."""
        logger.debug("asr_video_vlm_callback flag=%s", flag)
        if flag == 3:
            if not text:
                logger.info("asr_text_empty agent_id=%s event_id=%s", agent_id, event_id)
                return

            history = self.conversation.get_history(agent_id)
            logger.info(
                "llm_history agent_id=%s event_id=%s history=%s",
                agent_id,
                event_id,
                json.dumps(history, ensure_ascii=False),
            )
            # Route the request using the actual conversation history.
            route_result = self.scope_router.classify(
                text,
                session_id=agent_id,
                event_id=event_id,
                history=history,
            )
            logger.info(
                "scope_route_decided agent_id=%s event_id=%s result=%s",
                agent_id,
                event_id,
                route_result.to_dict(),
            )
            current_route = route_name(route_result.route)

            item_id = IdGenerator.generate_item_id()
            if route_result.route is ScopeRoute.TIMEOUT:
                timeout_reply = self.scope_router.config.timeout_reply_text
                response.on_interrupt(event_id, "chat", None)
                response.on_vlm_item_delta(
                    event_id,
                    item_id,
                    timeout_reply,
                )
                response.on_vlm_item_done(event_id, item_id)
                response.on_vlm_done(event_id)
                self.conversation.add_failed_turn(
                    agent_id,
                    user_text=text,
                    assistant_text=timeout_reply,
                )
                log_conversation(
                    agent_id=agent_id,
                    event_id=event_id,
                    user_text=text,
                    intent=current_route,
                    assistant_text=timeout_reply,
                )
                return

            if route_result.route is ScopeRoute.NOISE:
                logger.info("noise_input_ignored agent_id=%s event_id=%s", agent_id, event_id)
                log_conversation(
                    agent_id=agent_id,
                    event_id=event_id,
                    user_text=text,
                    intent=current_route,
                    assistant_text="",
                )
                return

            if route_result.route is ScopeRoute.STOP:
                logger.info("stop_intent_received agent_id=%s event_id=%s", agent_id, event_id)
                response.on_interrupt(event_id, "chat", None)
                log_conversation(
                    agent_id=agent_id,
                    event_id=event_id,
                    user_text=text,
                    intent=current_route,
                    assistant_text="",
                )
                return

            if self.tts_state == "playing":
                logger.info(
                    "request_ignored_while_speaking agent_id=%s event_id=%s route=%s",
                    agent_id,
                    event_id,
                    current_route,
                )
                log_conversation(
                    agent_id=agent_id,
                    event_id=event_id,
                    user_text=text,
                    intent=current_route,
                    assistant_text="",
                )
                return

            if route_result.route is ScopeRoute.SKILL:
                response.on_interrupt(event_id, "chat", None)
                skill_result = self.skill_handler.handle(
                    text,
                    history=history,
                    session_id=agent_id,
                    param=param,
                    event_id=event_id,
                    response=response,
                    item_id=item_id,
                )
                if skill_result.history_text:
                    self.conversation.add_turn(
                        agent_id,
                        user_text=text,
                        assistant_text=skill_result.history_text,
                    )
                else:
                    logger.error(
                        "skill_handler_returned_empty agent_id=%s event_id=%s",
                        agent_id,
                        event_id,
                    )
                log_conversation(
                    agent_id=agent_id,
                    event_id=event_id,
                    user_text=text,
                    intent=current_route,
                    assistant_text=skill_result.assistant_text,
                )
                return

            # Play an optional waiting reply, then dispatch the request.
            response.on_interrupt(event_id, "chat", None)
            llm_waiting_reply = build_llm_waiting_reply(
                get_llm_waiting_replies(
                    self.llm_waiting_replies,
                    current_route,
                )
            )
            if llm_waiting_reply:
                response.on_vlm_item_delta(event_id, item_id, llm_waiting_reply)
                response.on_vlm_item_done(event_id, item_id)

            item_id = IdGenerator.generate_item_id()
            reply_text = self.route_dispatcher.dispatch(
                route_result.route,
                HandlerContext(
                    user_text=text,
                    session_id=agent_id,
                    param=param,
                    event_id=event_id,
                    response=response,
                    item_id=item_id,
                    history=history,
                ),
            )
            if reply_text:
                self.conversation.add_turn(
                    agent_id,
                    user_text=text,
                    assistant_text=reply_text,
                )
                assistant_text = reply_text
            else:
                timeout_reply = self.llm_timeout_replies[current_route]
                response.on_vlm_item_delta(event_id, item_id, timeout_reply)
                response.on_vlm_item_done(event_id, item_id)
                response.on_vlm_done(event_id)
                self.conversation.add_failed_turn(
                    agent_id,
                    user_text=text,
                    assistant_text=timeout_reply,
                )
                assistant_text = timeout_reply
            spoken_text = "\n".join(
                part for part in (llm_waiting_reply, assistant_text) if part
            )
            log_conversation(
                agent_id=agent_id,
                event_id=event_id,
                user_text=text,
                intent=current_route,
                assistant_text=spoken_text,
            )

        else:
            logger.debug("non_text_frame_ignored flag=%s", flag)

    def on_face_info(self, agent_id, event_id, param: AgentParam):
        return None

    def on_video_frame(self, agent_id, event_id, flag, buf, is_key_frame, local_ts, param):
        return None

    def on_state(self, agent_id, event_id, state_name, state_value):
        logger.debug("agent_state_received state_name=%s", state_name)
        if state_name == "robot_pose":
            pose = json.loads(state_value)
            logger.info(
                "robot_pose x=%s y=%s yaw=%s",
                pose["x"],
                pose["y"],
                pose["yaw"],
            )
        elif state_name == "tts_status":
            tts = json.loads(state_value)
            tts_state = tts.get("state")
            if isinstance(tts_state, str) and tts_state:
                previous_tts_state = self.tts_state
                self.tts_state = tts_state
                logger.info(
                    "tts_state_changed previous=%s current=%s",
                    previous_tts_state,
                    self.tts_state,
                )

        elif state_name == "robot_form":
            # robot_form is a raw string and does not require json.loads.
            logger.info("robot_form value=%s", state_value)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logging.getLogger("linksoul_agentsdk.client").addFilter(
        _SdkStreamingDeltaLogFilter()
    )
    agent_config = load_agent_config("config.json")

    # Create the SDK singleton for this app_id.
    sdk = AgentSdk.create(
        url=agent_config.sdk_url,
        app_id=agent_config.app_id,
        app_key=agent_config.app_key,
        app_secret=agent_config.app_secret,
    )

    # Register the authentication-state callback.
    sdk.register_auth(MyAuth())

    # Register the passive ASR/video-to-VLM callback.
    callback = _Cb(sdk)
    sdk.register_asr_video2_vlm(callback)

    # Initialize the WebSocket connection and automatic reconnection.
    try:
        sdk.initialize()
        logger.info("application_initialized waiting_for_agent_auth=true")
        while True:
            time.sleep(3)
    except KeyboardInterrupt:
        logger.info("application_stopping")
    finally:
        try:
            callback.close()
        finally:
            sdk.release()


if __name__ == "__main__":
    main()
