import os
from pathlib import Path
basePath = cwd = Path(__file__).parent
filePath = os.path.join(basePath, 'saved_model/sss.a')
print(filePath)