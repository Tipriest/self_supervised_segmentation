import sys
import ast

def rgb_to_hex(rgb):
    # 将 RGB 元组转换为十六进制颜色码
    return '# 0x{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])

def main():
    if len(sys.argv) != 2:
        print("Usage: python rgbcode_conversion.py '<R, G, B>'")
        sys.exit(1)
    
    try:
        # 获取命令行输入的 RGB 元组字符串并转换为元组
        rgb_input = ast.literal_eval(sys.argv[1])
        
        # 检查输入是否为有效的 RGB 元组
        if len(rgb_input) != 3 or not all(0 <= value <= 255 for value in rgb_input):
            print("Error: Input must be a tuple of 3 integers, each between 0 and 255.")
            sys.exit(1)
        
        # 转换为十六进制颜色码
        hex_code = rgb_to_hex(rgb_input)
        print(hex_code)
    
    except (ValueError, SyntaxError):
        print("Error: Input must be a valid tuple of integers.")
        sys.exit(1)

if __name__ == "__main__":
    main()
    