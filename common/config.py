# -*- coding: utf-8 -*-
import os
import json
import datetime


import os
import json
import datetime

class JsonConfig(dict):
    """Load, merge and save JSON config files."""

    Indent = 2

    def __init__(self, *argv, **kwargs):
        super().__init__()
        super().__setitem__("__name", "default")

        assert len(argv) == 0 or len(kwargs) == 0, "Cannot use both args and kwargs."

        if len(argv) > 0:
            assert len(argv) == 1, "Only one positional arg allowed."
            arg = argv[0]
        else:
            arg = kwargs

        if isinstance(arg, str):
            super().__setitem__("__name", os.path.splitext(os.path.basename(arg))[0])
            with open(arg, "r") as load_f:
                arg = json.load(load_f)

        if isinstance(arg, dict):
            for key in arg:
                value = arg[key]
                if isinstance(value, dict):
                    value = JsonConfig(value)
                super().__setitem__(key, value)
        else:
            raise TypeError(f"Unsupported type: {type(arg)}")

    def __setattr__(self, attr, value):
        try:
            return super().__setitem__(attr, value)
        except KeyError:
            raise AttributeError(attr)

    def __getattr__(self, attr):
        try:
            return super().__getitem__(attr)
        except KeyError:
            raise AttributeError(attr)

    def __str__(self):
        return self.__to_string("", 0)

    def __to_string(self, name, intent):
        ret = " " * intent + name + " {\n"
        for key in self:
            if key.find("__") == 0:
                continue
            value = self[key]
            line = " " * intent
            if isinstance(value, JsonConfig):
                line += value.__to_string(key, intent + JsonConfig.Indent)
            else:
                line += " " * JsonConfig.Indent + key + ": " + str(value)
            ret += line + "\n"
        ret += " " * intent + "}"
        return ret

    def __add__(self, b):
        """
        魔术方法：重载 + 运算符，实现两个配置对象合并。
        若存在相同键，值必须一致或可递归合并；否则抛出异常。
        """
        assert isinstance(b, JsonConfig)
        for k in b:
            v = b[k]
            if k in self:
                if isinstance(v, JsonConfig):
                    # 子配置递归合并
                    super().__setitem__(k, self[k] + v)
                else:
                    if k == "__name":
                        super().__setitem__(k, self[k] + "&" + v)
                    else:
                        assert v == self[k], (
                            "[JsonConfig]: 配置冲突键 `{}`: {} != {}".format(k, self[k], v))
            else:
                # 新增键，直接添加
                super().__setitem__(k, v)
        return self

    def date_name(self):
        """
        生成基于当前时间和配置名的文件名，用于保存。
        """
        date = str(datetime.datetime.now())
        date = date[:date.rfind(":")].replace("-", "")\
                                     .replace(":", "")\
                                     .replace(" ", "_")
        return date + "_" + super().__getitem__("__name") + ".json"

    def dump(self, dir_path, json_name=None):
        if json_name is None:
            json_name = self.date_name()
        json_path = os.path.join(dir_path, json_name)
        with open(json_path, "w") as fout:
            json.dump(self.to_dict(), fout, indent=JsonConfig.Indent)

    def to_dict(self):
        ret = {}
        for k in self:
            if k.find("__") == 0:
                continue
            v = self[k]
            if isinstance(v, JsonConfig):
                ret[k] = v.to_dict()
            else:
                ret[k] = v
        return ret
