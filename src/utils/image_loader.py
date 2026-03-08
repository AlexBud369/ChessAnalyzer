import pygame
from pathlib import Path
from src.constants import PIECE_TYPES

def load_piece_images():
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent.parent
    images_dir = project_root / 'assets' / 'images'

    if not images_dir.exists():
        raise FileNotFoundError(f"Папка с изображениями не найдена: {images_dir}")

    images = {}
    for filename in PIECE_TYPES :
        for ext in ['.png', '.svg', '.jpg', '.jpeg', '.bmp']:
            file_path = images_dir / (filename + ext)
            if file_path.exists():
                try:
                    image = pygame.image.load(str(file_path)).convert_alpha()
                    images[filename] = image
                    break
                except pygame.error as e:
                    print(f"Не удалось загрузить {file_path}: {e}")
        else:
            print(f"Изображение для {filename} не найдено")
    return images