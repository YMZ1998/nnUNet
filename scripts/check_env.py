from dotenv import load_dotenv
import os

load_dotenv()  # 自动读取 .env 文件
print(os.environ['nnUNet_raw'])
