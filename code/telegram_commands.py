import subprocess
import traceback

from telegram import Update
from telegram.ext import ContextTypes

from village.classes.enums import State
from village.custom_classes.telegram_command_base import TelegramCommandBase
from village.manager import manager
from village.scripts.log import log

# "Mouse inside" == a task is running. Same set as check_box_lights().
TASK_RUNNING_STATES = [
    State.RUN_FIRST,
    State.CLOSE_DOOR2,
    State.OPEN_DOOR2,
    State.RUN_CLOSED,
    State.RUN_OPENED,
    State.SAVE_INSIDE,
    State.WAIT_EXIT,
    State.OPEN_DOOR2_STOP,
    State.RUN_MANUAL,
]


class IsThereAMouseInside(TelegramCommandBase):
    """/is_there_a_mouse_inside -> whether a task is currently running."""

    command = "is_there_a_mouse_inside"

    async def handler(self, update: Update,
                      context: ContextTypes.DEFAULT_TYPE) -> None:
        try:
            inside = manager.state in TASK_RUNNING_STATES
            answer = "yes" if inside else "no"
            await update.message.reply_text(
                f"{answer} ({manager.state.name}: {manager.state.description})"
            )
        except Exception:
            log.error("Telegram is_there_a_mouse_inside",
                      exception=traceback.format_exc())


class GiveIp(TelegramCommandBase):
    """/give_ip -> the LAN IP(s) of the Raspberry Pi."""

    command = "give_ip"

    async def handler(self, update: Update,
                      context: ContextTypes.DEFAULT_TYPE) -> None:
        try:
            ip = subprocess.check_output(["hostname", "-I"], text=True).strip()
            await update.message.reply_text(ip or "no IP found")
        except Exception:
            log.error("Telegram give_ip", exception=traceback.format_exc())


class RestartAnydesk(TelegramCommandBase):
    """/restart_anydesk -> restart the AnyDesk service on the Pi."""

    command = "restart_anydesk"

    async def handler(self, update: Update,
                      context: ContextTypes.DEFAULT_TYPE) -> None:
        try:
            subprocess.run(
                ["sudo", "systemctl", "restart", "anydesk"],
                check=True, capture_output=True, text=True, timeout=30,
            )
            await update.message.reply_text("anydesk restarted")
        except Exception:
            log.error("Telegram restart_anydesk",
                      exception=traceback.format_exc())
            await update.message.reply_text("failed to restart anydesk")
