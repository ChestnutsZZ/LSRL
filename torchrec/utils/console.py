"""命令行程序相关辅助函数"""
from argparse import ArgumentParser
from typing import Type

from global_utils.argument import IWithArguments


def add_console_arguments(parser: ArgumentParser, class_with_arguments: IWithArguments) -> ArgumentParser:
    """将对象携带的参数描述信息添加到命令行解析器"""
    for description in class_with_arguments.get_argument_descriptions():
        if description.default_value:
            parser.add_argument(
                f"--{description.name}",
                type=description.type_,
                required=description.is_required,
                help=description.help_info,
                default=description.default_value
            )
        else:
            parser.add_argument(
                f"--{description.name}",
                type=description.type_,
                required=description.is_required,
                help=description.help_info
            )
    return parser
