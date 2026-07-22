# Poseidon AI

> A modular AI platform for underwater computer vision, diver safety, marine intelligence, and ocean analytics.

---

## Overview

Poseidon AI is an open-source computer vision platform designed to build intelligent underwater AI systems.

The project aims to provide reusable, production-ready components for loading, validating, preprocessing, analyzing, and eventually detecting marine life from underwater imagery and video.

Rather than being a single machine learning model, Poseidon AI is designed as a collection of modular AI services that can evolve independently while working together as a unified platform.

---

## Current Module

### Nautilus Vision

Nautilus Vision is the first component of the Poseidon AI ecosystem.

It provides the foundational computer vision pipeline required before any AI model performs inference.

Current capabilities include:

- Image loading
- Image validation
- Metadata extraction
- Dataset loading
- Image preprocessing
- Command-line inspection tools
- Automated unit testing

---

## Long-Term Vision

The Poseidon AI platform will eventually consist of several specialized AI modules.

| Module | Purpose |
|---------|----------|
| Nautilus Vision | Computer Vision |
| BuddySense | Diver Tracking |
| CurrentAI | Ocean Current Prediction |
| DecoGuard | Decompression Safety |
| DiveAware | Dive Analytics |
| SurfaceOps | Monitoring Dashboard |

---

# Repository Structure

```text
poseidon-ai/

├── architecture/
├── configs/
├── data/
├── docs/
├── models/
├── notebooks/
├── scripts/
├── src/
│   └── poseidon_ai/
│       └── nautilus_vision/
├── tests/
├── pyproject.toml
└── README.md
```

---

# Installation

Clone the repository.

```bash
git clone https://github.com/JervisAnthony/poseidon-ai.git
```

Enter the project.

```bash
cd poseidon-ai
```

Create a virtual environment.

```bash
python -m venv .venv
```

Activate it.

Windows

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the package.

```bash
python -m pip install -e ".[dev]"
```

---

# Running Tests

Run all unit tests.

```bash
python -m pytest
```

---

# Command Line Tools

Inspect an image.

```bash
poseidon-inspect image.jpg
```

Display help.

```bash
poseidon-inspect --help
```

---

# Development Principles

Poseidon AI follows several engineering principles.

- Modular architecture
- Production-first design
- Test-driven development
- Clean package structure
- Reusable components
- Small, meaningful Git commits
- Comprehensive documentation

---

# Technology Stack

- Python 3.13
- OpenCV
- NumPy
- PyTest
- Setuptools
- Git
- GitHub

Future additions include:

- Ultralytics YOLO
- PyTorch
- FastAPI
- Docker
- Azure
- AWS
- Kubernetes

---

# Roadmap

## Foundation

- [x] Project packaging
- [x] Image loading
- [x] Metadata extraction
- [x] Image validation
- [x] Dataset loading
- [x] Image preprocessing
- [x] CLI tools
- [x] Unit testing

## Computer Vision

- [ ] YOLO model integration
- [ ] Marine life detection
- [ ] Video inference
- [ ] Live camera support

## AI Platform

- [ ] REST API
- [ ] Docker deployment
- [ ] CI/CD
- [ ] Cloud deployment
- [ ] MLOps
- [ ] Continuous model evaluation

---

# License

This project is licensed under the MIT License.