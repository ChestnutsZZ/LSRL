"""
将 url 编码文件转化为 utf8 编码，在命令行独立使用
参数格式：python url2utf8.py --input_file <filename> [--output_file <filename>]
"""

from argparse import ArgumentParser

from tqdm import tqdm

if __name__ == '__main__':
    parser: ArgumentParser = ArgumentParser(description='url 编码文件转换到 utf-8 编码文件')
    parser.add_argument('--input_file', type=str, required=True, help='输入文件名（ url 编码）')
    parser.add_argument('--output_file', type=str, default='', help='输出文件名（ utf-8 编码）')
    args, extras = parser.parse_known_args()

    if args.output_file == '':
        args.output_file = args.input_file + '.utf8'

    with open(args.input_file, encoding='ISO-8859-1') as in_f:
        with open(args.output_file, 'w') as out_f:
            for line in tqdm(in_f):
                out_f.write(line)

    print('转换结束')
