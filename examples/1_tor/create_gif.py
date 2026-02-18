#!/usr/bin/env python3
"""
Скрипт для создания GIF-анимации из PNG-кадров
"""

import os
import glob
from PIL import Image
import argparse


def create_gif(input_dir, output_file, duration=100, loop=0, optimize=True):
    """
    Создаёт GIF-анимацию из PNG-файлов в директории
    
    Параметры:
    ----------
    input_dir : str
        Путь к директории с PNG-файлами
    output_file : str
        Путь к выходному GIF-файлу
    duration : int
        Длительность каждого кадра в миллисекундах (по умолчанию 100мс = 10 FPS)
    loop : int
        Количество повторений (0 = бесконечно)
    optimize : bool
        Оптимизировать размер GIF (по умолчанию True)
    """
    
    # Находим все PNG-файлы и сортируем их
    pattern = os.path.join(input_dir, '*.png')
    image_files = sorted(glob.glob(pattern))
    
    if not image_files:
        print(f"❌ Не найдено PNG-файлов в директории: {input_dir}")
        return
    
    print(f"📁 Найдено {len(image_files)} кадров")
    print(f"📝 Первый кадр: {os.path.basename(image_files[0])}")
    print(f"📝 Последний кадр: {os.path.basename(image_files[-1])}")
    
    # Загружаем изображения
    images = []
    for i, img_path in enumerate(image_files):
        try:
            img = Image.open(img_path)
            images.append(img)
            if (i + 1) % 10 == 0:
                print(f"⏳ Загружено {i + 1}/{len(image_files)} кадров...")
        except Exception as e:
            print(f"⚠️  Ошибка при загрузке {img_path}: {e}")
    
    if not images:
        print("❌ Не удалось загрузить ни одного изображения")
        return
    
    # Создаём GIF
    print(f"\n🎬 Создание GIF-анимации...")
    print(f"   Длительность кадра: {duration}мс ({1000/duration:.1f} FPS)")
    print(f"   Повторений: {'∞' if loop == 0 else loop}")
    print(f"   Оптимизация: {'Да' if optimize else 'Нет'}")
    
    images[0].save(
        output_file,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=loop,
        optimize=optimize
    )
    
    # Информация о результате
    file_size = os.path.getsize(output_file) / (1024 * 1024)  # МБ
    print(f"\n✅ GIF создан успешно!")
    print(f"   Файл: {output_file}")
    print(f"   Размер: {file_size:.2f} МБ")
    print(f"   Кадров: {len(images)}")
    print(f"   Разрешение: {images[0].size[0]}x{images[0].size[1]}")


def main():
    parser = argparse.ArgumentParser(
        description='Создание GIF-анимации из PNG-кадров',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  # Базовое использование (по умолчанию frames_drag_test -> animation.gif)
  python create_gif.py
  
  # Указать свою директорию и выходной файл
  python create_gif.py -i frames_drag_test -o my_animation.gif
  
  # Изменить скорость (50мс = 20 FPS)
  python create_gif.py -d 50
  
  # Без оптимизации (больший размер, но быстрее создаётся)
  python create_gif.py --no-optimize
  
  # Ограничить количество повторений
  python create_gif.py -l 5
        """
    )
    
    parser.add_argument(
        '-i', '--input-dir',
        default='frames_torus_v2',
        help='Директория с PNG-файлами (по умолчанию: frames_drag_test)'
    )
    
    parser.add_argument(
        '-o', '--output',
        default='animation.gif',
        help='Имя выходного GIF-файла (по умолчанию: animation.gif)'
    )
    
    parser.add_argument(
        '-d', '--duration',
        type=int,
        default=100,
        help='Длительность кадра в миллисекундах (по умолчанию: 100мс = 10 FPS)'
    )
    
    parser.add_argument(
        '-l', '--loop',
        type=int,
        default=0,
        help='Количество повторений (0 = бесконечно, по умолчанию: 0)'
    )
    
    parser.add_argument(
        '--no-optimize',
        action='store_true',
        help='Отключить оптимизацию GIF (быстрее, но больше размер)'
    )
    
    args = parser.parse_args()
    
    # Получаем абсолютный путь к директории скрипта
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, args.input_dir)
    output_file = os.path.join(script_dir, args.output)
    
    print("=" * 60)
    print("🎨 GIF Creator")
    print("=" * 60)
    
    create_gif(
        input_dir=input_dir,
        output_file=output_file,
        duration=args.duration,
        loop=args.loop,
        optimize=not args.no_optimize
    )
    
    print("=" * 60)


if __name__ == '__main__':
    main()
