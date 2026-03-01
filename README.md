# Proyek Akhir: Membangun Sistem Machine Learning

Repositori ini berisi proyek akhir untuk kursus **Membangun Sistem Machine Learning** di **Dicoding**, yang dikerjakan oleh **Ahmad Ngiliyun (ahmadngiliyun00)**. Proyek ini memuat komponen utama terkait pengembangan dan deployment model machine learning, meliputi pipeline CI/CD, containerization dengan Docker, dan struktur direktori untuk tahap pemodelan dan monitoring.

## 📂 Struktur Proyek

- `Membangun_model/`: Berisi kode sumber untuk membangun dan melatih model machine learning.
- `Monitoring_dan_Logging/`: Berisi implementasi terkait sistem monitoring dan logging.
- `Eksperimen_SML_Ahmad-Ngiliyun.txt`: Tautan menuju repositori eksperimen dan preprocessing.
- `Workflow-CI.txt`: Detail dan konfigurasi workflow Continuous Integration (CI).
- `URL_DOCKER.txt`: Informasi terkait repositori DockerHub untuk proyek ini.

## 🔬 Repositori Eksperimen & Preprocessing (Kriteria 1)

Untuk proses eksperimen awal data dan tahap preprocessing, repositori terpisah dapat diakses pada tautan berikut:

- **GitHub**: [Eksperimen_SML_Ahmad-Ngiliyun](https://github.com/ahmadngiliyun00/Eksperimen_SML_Ahmad-Ngiliyun)

## 🐳 Docker Container

Proyek ini telah dikontainerisasi menggunakan Docker dan mendistribusikan image melalui DockerHub.

- **DockerHub Repository**: [workflow-ci-student-success](https://hub.docker.com/repository/docker/ahmadngiliyun00/workflow-ci-student-success/general)
- **Image Tag**: `ahmadngiliyun00/workflow-ci-student-success:latest`
- **Pull Command**:
  ```bash
  docker pull ahmadngiliyun00/workflow-ci-student-success:latest
  ```

> **Catatan untuk Reviewer**:
>
> - Image ini dibuat dari proses workflow CI (pipeline) dan di-push ke DockerHub.
> - Image ini saat ini tersedia untuk arsitektur (platform) `linux/amd64`.
> - Jika saat melakukan pull muncul pesan error seperti `"no matching manifest for linux/arm64/v8"`, hal ini terjadi karena environment Anda menggunakan arsitektur ARM (misal perangkat Apple Silicon/arm64). Silakan gunakan mesin atau runner berbasis `linux/amd64` untuk melakukan pull dan run image ini.

## ⚙️ Workflow & Continuous Integration (CI)

Proyek ini menggunakan GitHub Actions sebagai alat bantu Continuous Integration (CI) untuk memastikan integritas kode dan dependencies secara otomatis.

- **Trigger**: Dijalankan otomatis pada setiap _push_ atau _pull request_ ke branch `main`.
- **Tahapan CI**:
  1. Setup environment Python (versi 3.x)
  2. Install dependencies via `pip install -r requirements.txt`
  3. Lint/Format (_optional_): menggunakan `ruff` atau `flake8`
  4. Run basic check (misalnya menguji file menggunakan `python -m py_compile`)
- File konfigurasi workflow tersimpan pada repositori GitHub di path `.github/workflows/ci.yml`.

## 📄 Lisensi

Proyek ini menggunakan lisensi **MIT License**. Silakan merujuk ke file [LICENSE](LICENSE) untuk detail lebih lanjut.

---

_Dikembangkan oleh [Ahmad Ngiliyun (ahmadngiliyun00)](https://github.com/ahmadngiliyun00) untuk penyelesaian program kelas Dicoding._
