import logging
import os
import threading

logger = logging.getLogger(__name__)

env_content = ""

# Set DISABLE_DIAMOND=1 to skip remote config fetch entirely (recommended
# when running outside the alibaba intranet). Otherwise the fetch is run
# in a worker thread with a hard timeout so a stuck DNS/TCP handshake
# can't freeze startup — see README's "环境耦合" caveat.
_DIAMOND_TIMEOUT_S = float(os.environ.get("DIAMOND_TIMEOUT", "5"))

if os.environ.get("DISABLE_DIAMOND", "").lower() in ("1", "true", "yes"):
    logger.info("Diamond fetch disabled via DISABLE_DIAMOND env var")
else:
    def _fetch_diamond_config(out: list) -> None:
        try:
            import diamond_client as diamond  # noqa: F401
            from diamond_client import DEFAULT_GROUP_NAME  # noqa: F401
            from diamond_client import DiamondClient

            client = DiamondClient()
            content = client.get_config("amap-eval-service", "DEFAULT_GROUP")
            out.append(content or "")
        except Exception as exc:
            logger.warning(f"Diamond config fetch failed: {exc}")
            out.append("")

    _result: list = []
    _t = threading.Thread(target=_fetch_diamond_config, args=(_result,), daemon=True)
    _t.start()
    _t.join(timeout=_DIAMOND_TIMEOUT_S)

    if _t.is_alive():
        logger.warning(
            f"Diamond fetch exceeded {_DIAMOND_TIMEOUT_S}s, falling back to .env. "
            "Set DISABLE_DIAMOND=1 to skip this probe entirely."
        )
        env_content = ""
    else:
        env_content = _result[0] if _result else ""
