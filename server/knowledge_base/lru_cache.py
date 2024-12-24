import threading

from configs import CACHED_FILE_NUM


# lru cache , 保证线程安全, 简单实现，互斥所有操作
class ThreadSafeLRUCache:
    def __init__(self, capacity: int):
        self._capacity = capacity
        self._cache = {}
        self._order = []
        self._lock = threading.Lock()

    def contain(self, key):
        with self._lock:
            return key in self._cache

    def get(self, key: str):
        with self._lock:
            if key in self._cache:
                # 将访问的键移到队列末尾
                self._order.remove(key)
                self._order.append(key)
                return self._cache[key]
            else:
                return None

    def put(self, key: str, value):
        with self._lock:
            if key in self._cache:
                # 更新已存在的键值对
                self._cache[key] = value
                self._order.remove(key)
                self._order.append(key)
            else:
                if len(self._cache) >= self._capacity:
                    # 移除最近最少使用的键值对
                    oldest_key = self._order.pop(0)
                    del self._cache[oldest_key]
                self._cache[key] = value
                self._order.append(key)


longPdfCachePool = ThreadSafeLRUCache(CACHED_FILE_NUM)
