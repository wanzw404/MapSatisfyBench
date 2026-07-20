import logging
from pathlib import Path
from typing import Optional
logger = logging.getLogger(__name__)
class LifecycleManager:
    def __init__(self, online_file_path: Optional[str] = None):
        self.root_path = Path.home()
        self.online_file = self.root_path / (online_file_path or "app.online")
        self.has_been_online = False  #记录是否已经上线过
    
    def check(self) -> Optional[str]:
        """检查在线状态文件"""
        try:
            if self.online_file.exists():
                return self.online_file.read_text().strip()
            return None
        except Exception as e:
            logger.error(f"检查在线状态文件失败: {e}")
            return None
    
    def online(self) -> bool:
        """设置服务为在线状态"""
        try:
            self.online_file.write_text("success\n")
            self.has_been_online = True  # 标记已经上线过
            logger.info(f"服务已上线: {self.online_file}")
            return True
        except Exception as e:
            logger.error(f"设置在线状态失败: {e}")
            return False
# 全局实例
lifecycle = LifecycleManager()