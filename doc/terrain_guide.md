# Hướng dẫn Terrain trong Unitree RL Mjlab

---

## 1. Model train trên Flat có chạy được trên Rough không?

**Không tốt.** Policy học từ terrain nào thì quen với terrain đó:

- Train **Flat** → policy học đi trên mặt phẳng → **không biết xử lý bậc thang, dốc** → ngã ngay khi gặp rough
- Train **Rough** → policy học cả hai → **vẫn chạy được trên flat** (flat là trường hợp dễ nhất)

Thứ tự khuyến nghị: train flat trước để ổn định dáng đi, rồi fine-tune trên rough (hoặc train rough ngay với curriculum bắt đầu từ flat).

---

## 2. Default terrain hiện tại là gì?

```
velocity_env_cfg.py (base):  terrain_type = "generator"  →  ROUGH_TERRAINS_CFG
g1/env_cfgs.py flat:         terrain_type = "plane"       →  None (ghi đè)
g1/env_cfgs.py rough:        giữ nguyên base              →  ROUGH_TERRAINS_CFG + curriculum=True
```

**Kết luận:**
- `Mjlab-Velocity-Flat-Unitree-G1` → **mặt phẳng tuyệt đối** (`terrain_type="plane"`)
- `Mjlab-Velocity-Rough-Unitree-G1` → **7 loại terrain** (stairs, slope, wave, rough...) với curriculum

### Terrain mặc định trong `ROUGH_TERRAINS_CFG`

| Tên | Loại | Tỉ lệ | Tham số chính |
|-----|------|--------|---------------|
| `flat` | Phẳng | 20% | — |
| `pyramid_stairs` | Cầu thang lên | 20% | step_height 0–10cm |
| `pyramid_stairs_inv` | Cầu thang xuống | 20% | step_height 0–10cm |
| `hf_pyramid_slope` | Dốc lên | 10% | slope 0–1.0 rad |
| `hf_pyramid_slope_inv` | Dốc xuống | 10% | slope 0–1.0 rad |
| `random_rough` | Gồ ghề ngẫu nhiên | 10% | noise 2–10cm |
| `wave_terrain` | Sóng | 10% | amplitude 0–20cm |

Grid: **10 hàng × 20 cột**, mỗi ô 8×8m. Hàng 0 = dễ, hàng 9 = khó nhất.

---

## 3. Chuyển từ Flat sang Rough

Chỉ cần **đổi task ID**, không cần sửa code:

```bash
# Flat
python scripts/train.py Mjlab-Velocity-Flat-Unitree-G1 --env.scene.num-envs=4096

# Rough
python scripts/train.py Mjlab-Velocity-Rough-Unitree-G1 --env.scene.num-envs=4096
```

Fine-tune từ checkpoint flat sang rough:
```bash
python scripts/train.py Mjlab-Velocity-Rough-Unitree-G1 \
  --env.scene.num-envs=4096 \
  --agent.resume=True \
  --agent.load-run="g1_velocity" \
  --agent.load-checkpoint="model_10000.pt"
```

---

## 4. Thêm terrain tự custom

### Cách A — Chỉnh mix terrain hiện có (đơn giản nhất)

Không cần tạo file mới. Thêm hàm mới vào `mjlab/tasks/velocity/config/g1/env_cfgs.py`:

```python
def unitree_g1_stairs_only_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Chỉ cầu thang, không có terrain khác."""
    cfg = unitree_g1_rough_env_cfg(play=play)

    from dataclasses import replace
    import mjlab.terrains as terrain_gen

    cfg.scene.terrain.terrain_generator = replace(
        cfg.scene.terrain.terrain_generator,
        sub_terrains={
            "stairs_up": terrain_gen.BoxPyramidStairsTerrainCfg(
                proportion=0.5,
                step_height_range=(0.05, 0.25),
                step_width=0.3,
                platform_width=2.0,
            ),
            "stairs_down": terrain_gen.BoxInvertedPyramidStairsTerrainCfg(
                proportion=0.5,
                step_height_range=(0.05, 0.25),
                step_width=0.3,
                platform_width=2.0,
            ),
        }
    )
    return cfg
```

Đăng ký task trong `mjlab/tasks/velocity/config/g1/__init__.py`:

```python
from .env_cfgs import unitree_g1_stairs_only_env_cfg

register_mjlab_task(
    task_id="Mjlab-Velocity-Stairs-Unitree-G1",
    env_cfg=unitree_g1_stairs_only_env_cfg(),
    play_env_cfg=unitree_g1_stairs_only_env_cfg(play=True),
    rl_cfg=unitree_g1_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)
```

Train:
```bash
python scripts/train.py Mjlab-Velocity-Stairs-Unitree-G1 --env.scene.num-envs=4096
```

---

### Cách B — Tạo loại terrain hoàn toàn mới

Ví dụ: terrain có hố (pit).

**Bước 1:** Thêm class vào `mjlab/terrains/primitive_terrains.py`:

```python
@dataclass(kw_only=True)
class BoxPitTerrainCfg(SubTerrainCfg):
    """Terrain với hố ở giữa."""
    pit_depth_range: tuple[float, float] = (0.2, 0.5)
    pit_width: float = 1.0
    proportion: float = 0.1

    def function(self, difficulty, spec, rng):
        pit_depth = self.pit_depth_range[0] + difficulty * (
            self.pit_depth_range[1] - self.pit_depth_range[0]
        )
        body = spec.body("terrain")
        # Tạo platform bên trái hố
        left = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, ...)
        # Tạo platform bên phải hố
        right = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, ...)

        return TerrainOutput(
            origin=np.array([self.size[0] / 2, self.size[1] / 2, 0.0]),
            geometries=[
                TerrainGeometry(geom=left, color=np.array([0.5, 0.3, 0.1, 1.0])),
                TerrainGeometry(geom=right, color=np.array([0.5, 0.3, 0.1, 1.0])),
            ],
        )
```

**Bước 2:** Export trong `mjlab/terrains/__init__.py`:

```python
from .primitive_terrains import BoxPitTerrainCfg as BoxPitTerrainCfg
```

**Bước 3:** Dùng trong config:

```python
sub_terrains={
    "pit": terrain_gen.BoxPitTerrainCfg(
        proportion=0.3,
        pit_depth_range=(0.1, 0.4),
        pit_width=1.5,
    ),
    "stairs_up": terrain_gen.BoxPyramidStairsTerrainCfg(proportion=0.4, ...),
    "flat": terrain_gen.BoxFlatTerrainCfg(proportion=0.3),
}
```

---

## 5. Preview terrain trước khi train

Xem terrain không cần chạy toàn bộ training:

```bash
# Xem ROUGH_TERRAINS_CFG mặc định
python mjlab/terrains/config.py

# Xem terrain tùy chỉnh
python - <<'EOF'
from mjlab.terrains.config import ROUGH_TERRAINS_CFG
from mjlab.terrains.terrain_importer import TerrainImporter, TerrainImporterCfg
import mujoco.viewer

cfg = TerrainImporterCfg(terrain_type="generator", terrain_generator=ROUGH_TERRAINS_CFG)
terrain = TerrainImporter(cfg, device="cpu")
mujoco.viewer.launch(terrain.spec.compile())
EOF
```

---

## 6. Các loại terrain có sẵn

```
mjlab/terrains/
├── primitive_terrains.py
│   ├── BoxFlatTerrainCfg            — mặt phẳng
│   ├── BoxPyramidStairsTerrainCfg   — cầu thang lên (step_height_range, step_width)
│   ├── BoxInvertedPyramidStairsTerrainCfg — cầu thang xuống
│   └── BoxRandomGridTerrainCfg      — ô vuông ngẫu nhiên (grid_height_range)
│
└── heightfield_terrains.py
    ├── HfPyramidSlopedTerrainCfg    — dốc (slope_range, inverted=True/False)
    ├── HfRandomUniformTerrainCfg    — gồ ghề ngẫu nhiên (noise_range)
    └── HfWaveTerrainCfg             — sóng (amplitude_range, num_waves)
```

### Tham số TerrainGeneratorCfg quan trọng

| Param | Ý nghĩa | Gợi ý |
|-------|---------|-------|
| `num_rows` | Số mức độ khó | 10 |
| `num_cols` | Số cột terrain song song | 20 |
| `curriculum=True` | Hàng 0 dễ → hàng N khó | Luôn bật khi train |
| `max_init_terrain_level` | Robot bắt đầu ở hàng nào | 3–5 |
| `size` | Kích thước mỗi ô (m) | (8.0, 8.0) |
| `proportion` | Tỉ lệ xuất hiện mỗi loại | Tổng phải = 1.0 |

---

## Tóm tắt

```
Muốn gì?                           Làm gì?
──────────────────────────────────────────────────────────
Train rough terrain có sẵn     →  Đổi task ID sang Rough
Thay đổi mix/tỉ lệ terrain     →  Cách A: thêm hàm trong env_cfgs.py
Loại terrain hoàn toàn mới     →  Cách B: tạo class trong primitive/heightfield_terrains.py
Fine-tune flat → rough         →  Dùng --agent.resume + --agent.load-checkpoint
```