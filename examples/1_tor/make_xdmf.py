"""
make_xdmf.py
═══════════════════════════════════════════════════════════════════════════
Генерирует XDMF-файлы из Dedalus HDF5-снапшотов для просмотра 3D полей
в ParaView (волюметрическая визуализация, стримлайны, изоповерхности).

Запуск:
    cd /Users/premananda/Desktop/MyProjects/dedalus
    python examples/1_tor/make_xdmf.py

Результат:
    snapshots_tornado/snapshots_tornado_s1.xdmf  ← открыть в ParaView

В ParaView:
    File → Open → snapshots_tornado_s1.xdmf → Apply
    Coloring: выбрать поле (speed / vorticity_mag / buoyancy / pressure)
    Filters → Contour (изоповерхность) или Volume (объём)
═══════════════════════════════════════════════════════════════════════════
"""

import os
import glob
import h5py
import numpy as np

SNAP_DIR = "snapshots_tornado"


def write_xdmf(h5_path: str) -> str:
    """
    Строит XDMF для одного .h5 файла Dedalus-v3.
    Возвращает путь к созданному .xdmf файлу.
    """
    xdmf_path = h5_path.replace(".h5", ".xdmf")
    h5_basename = os.path.basename(h5_path)

    with h5py.File(h5_path, "r") as f:
        times = f["scales/sim_time"][:]          # (Nt,)
        # Dedalus 3 stores axes with hash suffixes: 'x_hash_<sha>', 'y_hash_<sha>', ...
        scale_keys = list(f["scales"].keys())
        x_key = next(k for k in scale_keys if k.startswith("x"))
        y_key = next(k for k in scale_keys if k.startswith("y"))
        z_key = next(k for k in scale_keys if k.startswith("z"))
        x     = f[f"scales/{x_key}"][:]             # (Nx,)
        y     = f[f"scales/{y_key}"][:]             # (Ny,)
        z     = f[f"scales/{z_key}"][:]             # (Nz,)
        tasks = list(f["tasks"].keys())
        # shapes: (Nt, Nx, Ny, Nz) for scalars, (Nt, 3, Nx, Ny, Nz) for vectors
        shapes = {k: f[f"tasks/{k}"].shape for k in tasks}

    Nx, Ny, Nz = len(x), len(y), len(z)
    Nt = len(times)

    # Скалярные поля (shape = (Nt, Nx, Ny, Nz))
    scalar_tasks = [k for k in tasks if len(shapes[k]) == 4]
    # Векторные поля пропускаем (ParaView requires component separation)

    lines = []
    lines.append('<?xml version="1.0" ?>')
    lines.append('<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>')
    lines.append('<Xdmf xmlns:xi="http://www.w3.org/2001/XInclude" Version="2.0">')
    lines.append('  <Domain>')
    lines.append('    <Grid Name="TimeSeries" GridType="Collection" CollectionType="Temporal">')

    for it, t in enumerate(times):
        lines.append(f'      <Grid Name="Frame_{it:04d}" GridType="Uniform">')
        lines.append(f'        <Time Value="{t:.6f}" />')
        # Топология: 3D структурированная сетка (Nz, Ny, Nx — порядок XDMF)
        lines.append(f'        <Topology TopologyType="3DRectMesh" Dimensions="{Nz} {Ny} {Nx}" />')
        # Геометрия: регулярные узлы (VXVYVZ)
        lines.append('        <Geometry GeometryType="VXVYVZ">')
        lines.append(f'          <DataItem Dimensions="{Nx}" NumberType="Float" Precision="8" Format="HDF">')
        lines.append(f'            {h5_basename}:/scales/{x_key}')
        lines.append('          </DataItem>')
        lines.append(f'          <DataItem Dimensions="{Ny}" NumberType="Float" Precision="8" Format="HDF">')
        lines.append(f'            {h5_basename}:/scales/{y_key}')
        lines.append('          </DataItem>')
        lines.append(f'          <DataItem Dimensions="{Nz}" NumberType="Float" Precision="8" Format="HDF">')
        lines.append(f'            {h5_basename}:/scales/{z_key}')
        lines.append('          </DataItem>')
        lines.append('        </Geometry>')

        # Добавляем каждое скалярное поле
        for task in scalar_tasks:
            lines.append(f'        <Attribute Name="{task}" AttributeType="Scalar" Center="Node">')
            lines.append(f'          <DataItem ItemType="HyperSlab" Dimensions="{Nz} {Ny} {Nx}" Type="HyperSlab">')
            lines.append('            <DataItem Dimensions="3 4" Format="XML">')
            # HyperSlab: [start] [stride] [count]
            # shape in h5: (Nt, Nx, Ny, Nz) → we want slice [it, :, :, :]
            lines.append(f'              {it} 0 0 0   1 1 1 1   1 {Nx} {Ny} {Nz}')
            lines.append('            </DataItem>')
            lines.append(f'            <DataItem Dimensions="{Nt} {Nx} {Ny} {Nz}" NumberType="Float" Precision="8" Format="HDF">')
            lines.append(f'              {h5_basename}:/tasks/{task}')
            lines.append('            </DataItem>')
            lines.append('          </DataItem>')
            lines.append('        </Attribute>')

        lines.append('      </Grid>')

    lines.append('    </Grid>')
    lines.append('  </Domain>')
    lines.append('</Xdmf>')

    with open(xdmf_path, "w") as out:
        out.write("\n".join(lines) + "\n")

    return xdmf_path


if __name__ == "__main__":
    h5_files = sorted(glob.glob(os.path.join(SNAP_DIR, "*.h5")))

    if not h5_files:
        print(f"❌ Нет .h5 файлов в '{SNAP_DIR}/'")
        print("   Сначала запустите: python tornado_3d.py")
        raise SystemExit(1)

    print(f"Найдено .h5 файлов: {len(h5_files)}")
    for h5 in h5_files:
        # Быстрая проверка содержимого
        with h5py.File(h5, "r") as f:
            times  = f["scales/sim_time"][:]
            tasks  = list(f["tasks"].keys())
            shapes = {k: f[f"tasks/{k}"].shape for k in tasks}
        print(f"\n  Файл : {h5}")
        print(f"  Кадры: {len(times)}  (t = {times[0]:.3f} … {times[-1]:.3f})")
        print(f"  Поля : {', '.join(tasks)}")
        for k, sh in shapes.items():
            print(f"          {k}: {sh}")

        xdmf = write_xdmf(h5)
        print(f"\n  ✓ XDMF создан: {xdmf}")

    print("\n══════════════════════════════════════════════════════")
    print("  Как открыть в ParaView:")
    print("  1. File → Open → snapshots_tornado_s1.xdmf")
    print("  2. В диалоге выберите 'Xdmf3ReaderS' → OK")
    print("  3. Нажмите кнопку 'Apply'")
    print("  4. В панели 'Coloring' выберите поле:")
    print("       speed          → скорость |u|")
    print("       vorticity_mag  → завихрённость |ω|")
    print("       buoyancy       → плавучесть / тепло")
    print("       pressure       → давление")
    print("  5. Filters → Contour — для изоповерхностей")
    print("     Filters → StreamTracer — для линий тока")
    print("     Filters → Slice — для горизонтальных срезов")
    print("  6. Play ▶ для анимации по времени")
    print("══════════════════════════════════════════════════════")
