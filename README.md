
# StellarNet 🌟

[![Join our Discord](https://img.shields.io/badge/Discord-Join%20our%20server-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/agora-999382051935506503) [![Subscribe on YouTube](https://img.shields.io/badge/YouTube-Subscribe-red?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@kyegomez3242) [![Connect on LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/kye-g-38759a207/) [![Follow on X.com](https://img.shields.io/badge/X.com-Follow-1DA1F2?style=for-the-badge&logo=x&logoColor=white)](https://x.com/kyegomezb)

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2024.xxxxx-b31b1b.svg)](https://arxiv.org/)


StellarNet: An AI system probing the possibility that stars may possess primitive forms of information processing. By analyzing complex patterns in stellar emissions using deep learning, we search for signatures of self-organization and structured behavior that transcend random processes.


## Overview

This project implements a comprehensive analysis pipeline for investigating potential "consciousness-like" patterns in stellar data using PyTorch and astronomical data from TESS and Kepler missions.


## Features

- 🔬 Real-time analysis of stellar light curves from TESS/Kepler missions
- 🧠 LSTM-based pattern detection for stellar behavior prediction
- 📊 Comprehensive entropy and frequency analysis
- 🔍 Anomaly detection in stellar emissions
- 📈 Advanced visualization of stellar patterns

## Installation

```bash
# Clone the repository
git clone https://github.com/Agora-Lab-AI/StellarNet.git
cd StellarNet

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```bash
python main.py
```

By default, the script analyzes a set of pre-selected variable stars. To analyze specific stars:

```bash
python main.py --star_id "TIC 260128333" --mission "TESS"
```


## Requirements

- Python 3.10+
- PyTorch
- lightkurve
- astropy
- numpy
- pandas
- scipy
- scikit-learn
- matplotlib

See `requirements.txt` for complete list.

## Methodology

Our analysis pipeline consists of several key components:

1. **Data Collection**: Automated fetching of stellar light curves from TESS/Kepler missions
2. **Preprocessing**: Cleaning and normalization of time-series data
3. **Pattern Analysis**:
   - Shannon entropy calculation
   - Fourier analysis
   - LSTM-based pattern prediction
   - Anomaly detection
4. **Visualization**: Comprehensive plotting of results

## Results

Analysis results are saved in the `results/` directory with the following structure:
- `{star_id}_analysis.npz`: Numerical results and statistics
- `{star_id}_plots.png`: Visualization plots
- `models/{star_id}_model.pt`: Trained LSTM model

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## Citation

If you use this code in your research, please cite:

```bibtex
@article{stellarnet2024,
  title={StellarNet: Investigating Information Processing Patterns in Stellar Emissions},
  author={Agora Lab AI, Kye Gomez},
  journal={arXiv preprint arXiv:2024.xxxxx},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- NASA's TESS and Kepler missions for providing stellar data
- The lightkurve team for their excellent data access tools
- The astropy community for their comprehensive astronomy tools

## Contact

- **Website**: [https://agoralab.ai](https://agoralab.ai)
- **Issues**: [GitHub Issues](https://github.com/Agora-Lab-AI/StellarNet/issues)
- Twitter: [@kyegomez](https://twitter.com/kyegomez)
- Email: kye@swarms.world

---

## Want Real-Time Assistance?

[Book a call with here for real-time assistance:](https://cal.com/swarms/swarms-onboarding-session)

---

⭐ Star us on GitHub if this project helped you!
