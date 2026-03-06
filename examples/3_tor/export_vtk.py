"""
export_vtk.py
═══════════════════════════════════════════════════════════════════════
Скрипт для конвертации HDF5 файлов (snapshots) в формат VTK (VTI + PVD)
для последующей визуализации в ParaView.

ЗАПУСК:
  python3 export_vtk.py
═══════════════════════════════════════════════════════════════════════
"""
import os
import glob
import h5py
import numpy as np
import logging
import argparse

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ── Параметры геометрии (те же, что и в основном скрипте) ────────────
Lx = Ly = Lz = 4 * np.pi

# ── Папки вывода ──────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SNAP_DIR   = os.path.join(SCRIPT_DIR, "snapshots_rotating_tori")
VTK_DIR    = os.path.join(SCRIPT_DIR, "vtk_output_rotating_tori")

def find_scale(f, name):
    """Найти шкалу (координату) в f['scales'] по имени x, y или z."""
    for key in f['scales'].keys():
        if key.startswith(f"{name}_hash_"):
            return f['scales'][key][:]
    return None

def export_vtk(h5_files, vtk_dir=VTK_DIR, default_lx=Lx, default_ly=Ly, default_lz=Lz):
    try:
        import pyvista as pv
    except ImportError:
        logger.error("pyvista не установлена. Установите: pip install pyvista")
        return

    pvd_name = "simulation_data.pvd"
    pvd_path = os.path.join(vtk_dir, pvd_name)
    os.makedirs(vtk_dir, exist_ok=True)
    
    if not h5_files:
        logger.warning(f"Нет файлов для экспорта")
        return

    pvd_entries = []
    fc = 0
    logger.info(f"Начинаю VTK-экспорт файлов ({len(h5_files)} шт.) → {vtk_dir}/")

    for fp in h5_files:
        logger.info(f"Обработка файла: {os.path.basename(fp)}")
        with h5py.File(fp, 'r') as f:
            times = f['scales/sim_time'][:]
            
            # ── Определение геометрии ──────────────────────────────────
            # Пробуем найти реальные размеры из шкал h5
            x_scale = find_scale(f, 'x')
            y_scale = find_scale(f, 'y')
            z_scale = find_scale(f, 'z')
            
            if x_scale is not None and y_scale is not None and z_scale is not None:
                Nx_, Ny_, Nz_ = len(x_scale), len(y_scale), len(z_scale)
                dx = x_scale[1] - x_scale[0] if Nx_ > 1 else 1.0
                dy = y_scale[1] - y_scale[0] if Ny_ > 1 else 1.0
                dz = z_scale[1] - z_scale[0] if Nz_ > 1 else 1.0
                origin = (x_scale[0], y_scale[0], z_scale[0])
            else:
                # Если шкалы не найдены, используем дефолтные параметры (fallback)
                logger.warning("Координатные шкалы не найдены, использую параметры геометрии по умолчанию.")
                first_task = list(f['tasks'].keys())[0]
                shape = f['tasks'][first_task].shape
                # Dedalus h5 shape: (time, x, y, z) или (time, component, x, y, z)
                if len(shape) == 4:
                    _, Nx_, Ny_, Nz_ = shape
                else:
                    _, dim, Nx_, Ny_, Nz_ = shape
                dx, dy, dz = default_lx / Nx_, default_ly / Ny_, default_lz / Nz_
                origin = (-default_lx/2, -default_ly/2, -default_lz/2)

            for i in range(len(times)):
                tv = float(times[i])
                grid = pv.ImageData()
                grid.dimensions = (Nx_, Ny_, Nz_)
                grid.spacing = (dx, dy, dz)
                grid.origin = origin
                
                # ── Экспорт ВСЕХ доступных полей (tasks) ────────────
                for task_name in f['tasks'].keys():
                    data = f['tasks'][task_name][i]
                    if len(data.shape) == 3:
                        # Скалярное поле
                        grid.point_data[task_name] = data.flatten(order="F")
                    elif len(data.shape) == 4:
                        # Векторное поле (3 компонента)
                        grid.point_data[task_name] = np.stack([
                            data[0].flatten(order="F"),
                            data[1].flatten(order="F"),
                            data[2].flatten(order="F"),
                        ], axis=1)

                # ── Специальный случай: если есть velocity, но нет speed ──
                if 'velocity' in f['tasks'] and 'speed' not in f['tasks']:
                    uv = f['tasks/velocity'][i]
                    speed_data = np.sqrt(uv[0]**2 + uv[1]**2 + uv[2]**2)
                    grid.point_data["speed"] = speed_data.flatten(order="F")
                
                vn = f"frame_{fc:04d}.vti"
                grid.save(os.path.join(vtk_dir, vn))
                pvd_entries.append((tv, vn))
                fc += 1

    # Запись PVD коллекции кадров
    with open(pvd_path, "w", encoding="utf-8") as pvd:
        pvd.write('<?xml version="1.0"?>\n')
        pvd.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
        pvd.write('  <Collection>\n')
        for te, vn in pvd_entries:
            pvd.write(f'    <DataSet timestep="{te:.6f}" group="" part="0" file="{vn}"/>\n')
        pvd.write('  </Collection>\n')
        pvd.write('</VTKFile>\n')
        
    logger.info(f"✅  VTK экспорт завершен: {fc} кадров → {pvd_path}")

def main():
    parser = argparse.ArgumentParser(description="Конвертация HDF5 в VTK (VTI + PVD) для ParaView.")
    parser.add_argument("input_path", nargs="?", default=SNAP_DIR,
                        help="Путь к папке со снапшотами (или конкретному .h5 файлу)")
    parser.add_argument("--out", "-o", dest="out_dir", default=None,
                        help="Папка для сохранения VTK файлов")
    
    args = parser.parse_args()

    input_path = args.input_path
    
    # Определяем файлы для обработки
    if os.path.isfile(input_path) and input_path.endswith('.h5'):
        # Если передан один файл
        snap_dir = os.path.dirname(input_path) or '.'
        h5_files = [input_path]
    elif os.path.isdir(input_path):
        # Если передана папка
        snap_dir = input_path
        h5_files = sorted(glob.glob(os.path.join(snap_dir, "*.h5")))
    else:
        logger.error(f"Указанный путь не найден или имеет неверный формат: {input_path}")
        return

    # Директория для вывода
    vtk_dir = args.out_dir if args.out_dir else VTK_DIR
    
    # Передаем найденные файлы вместо snap_dir для проверки внутри
    if not h5_files:
        logger.warning(f"Нет .h5 файлов для экспорта по пути: {input_path}")
        return
        
    # Вызываем измененную функцию
    if args.out_dir is None:
        # Если папка вывода не указана, выведем в подпапку vtk_output внутри папки с h5
        vtk_dir = os.path.join(snap_dir, "vtk_output")
    else:
        vtk_dir = args.out_dir
        
    export_vtk(h5_files, vtk_dir=vtk_dir)

if __name__ == '__main__':
    main()
