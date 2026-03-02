"""
convert_to_vtk.py
═══════════════════════════════════════════════════════════════════════
Конвертирует HDF5-снапшоты GPE-симуляции в формат VTK для ParaView.

ВЫВОД:
  vtk_output_gpe_opposite_spin/
    ├── gpe_frame_0000.vti      ← 3D-кадры (ImageData)
    ├── gpe_frame_0001.vti
    ├── ...
    └── gpe_simulation.pvd      ← коллекция с временны́ми метками
                                   (открывать в ParaView именно этот файл)

ПОЛЯ В КАЖДОМ КАДРЕ:
  density   — плотность |ψ|²  (основное поле для визуализации)
  psi_real  — Re(ψ)
  psi_imag  — Im(ψ)
  phase     — фаза arg(ψ) ∈ (−π, π]  (выявляет топологические вихри)

ЗАПУСК:
  /Users/premananda/miniforge3/envs/dedalus_env/bin/python examples/1_tor/convert_to_vtk.py
═══════════════════════════════════════════════════════════════════════
"""
import os
import glob
import h5py
import numpy as np

try:
    import pyvista as pv
except ImportError:
    print("Ошибка: pyvista не установлена в текущем окружении.")
    print("Установите: /Users/premananda/miniforge3/envs/dedalus_env/bin/pip install pyvista")
    exit(1)

# ── Папки ─────────────────────────────────────────────────────────
input_dir  = "examples/1_tor/snapshots_gpe_opposite_spin"
output_dir = "examples/1_tor/vtk_output_gpe_opposite_spin"
pvd_name   = "gpe_simulation.pvd"   # главный файл для открытия в ParaView

os.makedirs(output_dir, exist_ok=True)

# ── Поиск .h5 файлов ──────────────────────────────────────────────
h5_files = sorted(glob.glob(os.path.join(input_dir, "*.h5")))

if not h5_files:
    print(f"Файлы .h5 не найдены в папке '{input_dir}'.")
    print("Убедитесь, что GPE-симуляция уже запущена и сохранила хотя бы один кадр.")
    exit(1)

print(f"Найдено .h5 файлов: {len(h5_files)}")
print(f"Конвертация в VTK + сборка PVD-коллекции для ParaView...\n")

# ── Параметры домена (совпадают с gpe_two_rings_opposite_spin.py) ──
Lx, Ly, Lz = 24.0, 24.0, 24.0

# Список записей для .pvd: (sim_time, relative_vti_path)
pvd_entries = []
frame_counter = 0

for file_path in h5_files:
    print(f"Читаем: {file_path}")
    with h5py.File(file_path, 'r') as f:
        times   = f['scales/sim_time'][:]

        # Определяем размер сетки из первого поля
        dens_ds = f['tasks/density']
        _, Nx, Ny, Nz = dens_ds.shape   # (time, x, y, z)

        dx = Lx / Nx
        dy = Ly / Ny
        dz = Lz / Nz

        for i in range(len(times)):
            t = float(times[i])

            # ── Читаем поля ───────────────────────────────────────
            density  = np.real(f['tasks/density'][i])    # |ψ|²
            psi_real = np.real(f['tasks/psi_real'][i])   # Re(ψ)
            psi_imag = np.real(f['tasks/psi_imag'][i])   # Im(ψ)

            # ── Создаём 3D ImageData-сетку ────────────────────────
            grid = pv.ImageData()
            grid.dimensions = (Nx, Ny, Nz)
            grid.spacing    = (dx, dy, dz)
            grid.origin     = (-Lx / 2, -Ly / 2, -Lz / 2)

            # Порядок Fortran: x изменяется быстрее всего (VTK-стандарт)
            grid.point_data["density"]  = density.flatten(order="F")
            grid.point_data["psi_real"] = psi_real.flatten(order="F")
            grid.point_data["psi_imag"] = psi_imag.flatten(order="F")
            grid.point_data["phase"]    = np.angle(
                psi_real + 1j * psi_imag
            ).flatten(order="F")

            # ── Сохраняем .vti кадр ───────────────────────────────
            vti_filename = f"gpe_frame_{frame_counter:04d}.vti"
            vti_path     = os.path.join(output_dir, vti_filename)
            grid.save(vti_path)
            print(f"  [{frame_counter:04d}] t={t:.3f}  →  {vti_filename}")

            # Запоминаем для PVD (путь относительно .pvd файла)
            pvd_entries.append((t, vti_filename))
            frame_counter += 1

# ── Генерируем .pvd файл ──────────────────────────────────────────
pvd_path = os.path.join(output_dir, pvd_name)
with open(pvd_path, "w", encoding="utf-8") as pvd:
    pvd.write('<?xml version="1.0"?>\n')
    pvd.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
    pvd.write('  <Collection>\n')
    for t, vti_filename in pvd_entries:
        pvd.write(f'    <DataSet timestep="{t:.6f}" group="" part="0" file="{vti_filename}"/>\n')
    pvd.write('  </Collection>\n')
    pvd.write('</VTKFile>\n')

print(f"\n{'═'*60}")
print(f"✅  Готово! Сконвертировано кадров: {frame_counter}")
print(f"📂  Папка с кадрами : {output_dir}/")
print(f"🎬  Файл для ParaView: {pvd_path}")
print(f"{'═'*60}")
print(f"\nОткройте в ParaView:")
print(f"  File → Open → {pvd_path}")
print(f"  Нажмите Apply, затем используйте Play или ползунок времени.")
print(f"\n  Рекомендуемые фильтры для визуализации вихрей:")
print(f"    • Contour → density  (isosurface плотности)")
print(f"    • Volume Rendering → density")
print(f"    • Contour → phase    (±π граница = вихревая нить)")