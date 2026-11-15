<div align="center">

# gazebo_sim

World Gazebo (arena maze + obstacle ArUco) beserta drone Iris dengan **6 LiDAR** dan **2 kamera**, untuk simulasi ArduPilot SITL + ROS 2 + MAVROS.

</div>

---

## Sensor drone

| Sensor | Jumlah | Tipe pesan |
|---|---|---|
| LiDAR | 6 | `sensor_msgs/LaserScan` |
| Kamera | 2 | `sensor_msgs/Image` |

Pada `LaserScan`, jarak ada di `ranges[0]` (meter):

| Nilai | Artinya |
|---|---|
| angka biasa | jarak ke benda (meter) |
| `inf` | tidak ada benda dalam 10 m |
| `-inf` | terlalu dekat (< 0,12 m) |

> Nama topik sensor **cari sendiri** setelah T3 berjalan (lihat bagian T3).

---

##  Cara menjalankan

Gunakan **3 terminal** secara berurutan.


```text
T1  Randomizer + Gazebo     (world & target)
T2  SITL ArduCopter         (flight controller)
T3  Bridge sensor + MAVROS  (data sensor & kontrol)
```

### T1: GAZEBOOO

Jalankan randomizer **dulu** untuk mendapatkan target (target disimpan di `target.json` kalau mau dilihat), baru jalankan Gazebo setelah itu.

```bash
cd ~/oprec/gazebo_sim/worlds/world_oprec
python3 randomize.py
```

```bash
gz sim -r world_oprec.world
```

### T2: SITL
jalankan environmentnya dulu 
```bash
ardu
sim_vehicle.py -v ArduCopter -f gazebo-iris --model JSON --console
```

Tunggu sampai muncul `EKF3 IMU origin set` dan status di console berwarna **hijau**. Kalau belum, drone tidak akan mau takeoff.

### T3: Bridge sensor + MAVROS

```bash
ros2 launch ~/oprec/gazebo_sim/scripts/sensors_mavros.launch.py
```

Script ini menjalankan MAVROS dan bridge Gazebo → ROS 2 untuk mendapatkan topik sensor (LiDAR dan kamera).

Cek sensor sudah masuk:

```bash
ros2 topic list                       # semua topik
ros2 topic list -t                    # beserta tipenya
ros2 topic list | grep -i lidar       # saring kata kunci
ros2 topic echo <nama_topik> --once   # lihat isi satu pesan
ros2 topic hz <nama_topik>            # laju update
```

---

##  Ketentuan & penilaian

**Aturan**

1. Drone **tidak boleh melewati batas atas dinding samping** pada maze.
2. Di bagian belakang ada **Bulls Eye** dengan pengali poin, tergantung bagian mana yang terkena:

| Mengenai | Pengali poin |
|---|---|
|  Tengah | × 1.0 |
| Ring ke-2 | × 0.8 |
| Ring luar | × 0.5 |
| Tidak mengenai | × 0.2 |

---

## Masalah umum

| Gejala | Penyebab / solusi |
|---|---|
| `mavproxy.py: No such file or directory` | Venv ArduPilot tidak aktif saat menjalankan `sim_vehicle.py`. Aktifkan venv hanya untuk T2. |
| `No JSON sensor message received` | Simulasi masih paused. Jalankan Gazebo dengan `-r`, dan hentikan proses lama: `pkill -f "gz sim"`. |
| `ros2 run ... executable_name required` | Pakai `ros2 launch <path-file>`, bukan `ros2 run`. |
| Drone tidak mau takeoff | Tunggu `EKF3 IMU origin set` di T2 (status hijau) dulu. |
| Topik sensor tidak muncul | Pastikan T3 sudah jalan, lalu `ros2 topic list`. |
| Susunan target tidak berubah | Tutup Gazebo, jalankan `randomizer.py` lagi, baru `gz sim`. |

---

## Catatan

- Tinggi dinding arena sekitar **2,2 m** (terukur dari model).
- Port MAVROS ke SITL: `udp://127.0.0.1:14550@`.
- Randomizer harus dijalankan **sebelum** Gazebo, karena Gazebo membaca model saat start.
