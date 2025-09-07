import importlib
import pkgutil
from typing import Dict, Type

REGISTRY: Dict[str, Type] = {}

def discover_and_register(
    root_package: str,
    target_filename: str = "adaptor.py",
    class_name: str = "Adaptor",
    base_class: type | None = None,
) -> Dict[str, Type]:
    """
    在 root_package 下递归查找所有子模块，定位文件名为 target_filename 的模块，
    导入后获取名为 class_name 的类（可选校验是否是 base_class 的子类），
    并注册到 REGISTRY，键名默认取该模块的上级包名或 __plugin_name__。

    返回 REGISTRY（便于链式使用或测试）。
    """
    pkg = importlib.import_module(root_package)
    target_mod = target_filename[:-3] if target_filename.endswith(".py") else target_filename

    if not hasattr(pkg, "__path__"):
        raise ValueError(f"{root_package!r} 不是一个包（没有 __path__）。")

    for _, modname, _ in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
        # 只处理文件名为 target_mod 的模块
        if modname.rsplit(".", 1)[-1] != target_mod:
            continue

        module = importlib.import_module(modname)
        cls = getattr(module, class_name, None)
        if cls is None:
            continue
        if base_class is not None and not issubclass(cls, base_class):
            continue

        # 取注册名：优先用模块里自定义的 __plugin_name__；否则用上级包名
        parent_pkg = module.__package__.rsplit(".", 1)[-1] if module.__package__ else modname
        name = getattr(module, "__plugin_name__", parent_pkg)

        REGISTRY[name] = cls

    return REGISTRY
