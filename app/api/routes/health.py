import os
import logging
from fastapi import APIRouter
from fastapi.responses import PlainTextResponse
from fastapi import HTTPException
from app.lifecycle.manager import lifecycle

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get('/status.taobao', response_class=PlainTextResponse)
def status_taobao():
    """
    健康检查接口
    - 如果是首次请求，自动上线
    - 如果已上线，返回状态
    - 如果状态文件不存在，返回404
    使用一个上线标志文件，主要是为了VIPServer优雅上下线，在停止应用前，首先会删除app.online文件，使VIPServer摘流
    """
    try:
        # 检查当前状态文件
        result = lifecycle.check()

        # 如果状态文件不存在，且没有上线过，说明是首次请求，自动上线
        if not result and not lifecycle.has_been_online:
            if lifecycle.online():
                return "success"
            else:
                raise HTTPException(status_code=500, detail="Failed to online")
        else:
            if result:
                # 已经上线，返回success
                return result
            else:
                # 状态文件不存在，说明app.online已被删除，返回404以便VIPServer摘流
                raise HTTPException(status_code=404, detail="not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"健康检查异常: {e}")
        raise HTTPException(status_code=404, detail="not found")


@router.get('/hello-world')
def hello():
    """
    Hellow World
    """
    env_vars = os.environ
    print('env vars:')
    for key, value in env_vars.items():
        print(f'{key}={value}')
    return f"Hello, World!"
