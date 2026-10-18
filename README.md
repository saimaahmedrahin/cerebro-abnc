# ABNC: Active Brain Noise Cancellation

**Portable EEG Denoising Algorithm for Early Dementia Screening**

![ABNC Algorithm Performance](docs/images/abnc_waveforms.png)

## Overview

ABNC (Active Brain Noise Cancellation) is a signal processing algorithm designed to remove noise from portable EEG recordings while preserving diagnostic frequencies. It targets early dementia screening by maintaining the integrity of slow-wave brain activity (1-4 Hz) while suppressing motion artifacts and powerline interference.

### Key Features

- **Real-time Processing**: 5 ms per 64-second EEG clip
- **Motion Artifact Removal**: Suppresses low-frequency movement noise (1-2 Hz)
- **Powerline Filtering**: Notch filter for 50/60 Hz interference
- **Diagnostic Preservation**: 96.2% preservation of diagnostic frequencies (1-4 Hz slow-waves)
- **SNR Improvement**: +14.4 dB average improvement over baseline

## Algorithm

### Architecture

ABNC combines three signal processing techniques:

1. **Bandpass Filter (1-4 Hz)**
   - Isolates diagnostic slow-wave frequencies
   - Uses 4th-order Butterworth filter
   - Preserves delta (1-4 Hz) activity associated with cognitive impairment

2. **Notch Filter (49-51 Hz)**
   - Removes 50 Hz powerline interference (60 Hz in North America)
   - 2nd-order Butterworth bandstop filter
   - Critical for portable/non-shielded environments

3. **Adaptive Wiener Filter**
   - Noise suppression while maintaining signal structure
   - 1-second window adaptation
   - Weighted combination: 70% bandpass + 30% Wiener (empirically optimized)

### Mathematical Foundation

```
Filtered Signal = 0.7 × BandPass(1-4 Hz) + 0.3 × WienerFilter(Corrupted)
```

Where:
- **BandPass** isolates diagnostic frequencies
- **WienerFilter** suppresses broadband noise (white noise, motion artifacts)
- Weighted combination balances frequency preservation vs. noise suppression

## Performance Metrics

### Validation Results (100 test samples, 4-channel EEG)

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **SNR (dB)** | -2.98 ± 0.03 | 11.45 ± 0.12 | **+14.43 dB** |
| **Diagnostic Preservation (1-4 Hz)** | — | 96.2% | — |
| **Processing Time (ms/clip)** | — | 5 | Real-time ✓ |
| **Powerline Suppression (50 Hz)** | — | -40 dB | Excellent ✓ |

### Frequency Domain Analysis

![Frequency Response](docs/images/abnc_frequency.png)

**Key observations:**
- Green shaded region (1-4 Hz): Diagnostic frequency band preserved
- Red dashed line (50 Hz): Powerline completely suppressed
- Baseline noise floor reduced by ~40 dB across 5-40 Hz band

## Installation

### Requirements

- Python 3.8+
- NumPy
- SciPy
- Matplotlib (for visualization)

### Setup

```bash
git clone https://github.com/saimaahmedrahin/cerebro-abnc.git
cd cerebro-abnc
pip install -r requirements.txt
```

## Usage

### Basic Example

```python
from abnc_algorithm import abnc_denoise
import numpy as np

# Load your EEG data (samples × channels × timepoints)
eeg_corrupted = np.load('your_eeg_file.npy')

# Apply ABNC denoising
eeg_cleaned = np.zeros_like(eeg_corrupted)

for ch in range(eeg_corrupted.shape[1]):
    eeg_cleaned[:, ch, :] = abnc_denoise(eeg_corrupted[:, ch, :], sample_rate=250)

# Result: Clean EEG with diagnostic frequencies preserved
print(f"Cleaned EEG shape: {eeg_cleaned.shape}")
```

### Parameters

```python
abnc_denoise(
    corrupted_signal,           # 1D array of EEG samples
    sample_rate=250,            # Sampling rate in Hz
    diagnostic_band=[1, 4],     # Frequency band to preserve (Hz)
    powerline_freq=50           # Powerline frequency (50 or 60 Hz)
)
```

### Full Pipeline Example

See `examples/` folder for complete walkthrough:
1. Data generation
2. SNR computation (before/after)
3. Frequency domain analysis
4. Visualization

## Clinical Application

### Use Case: Early Dementia Screening

Slow-wave EEG activity (1-4 Hz) is a biomarker for cognitive decline. ABNC enables reliable detection using:

- **Portable EEG hardware** (dry electrodes, low-cost sensors)
- **Non-clinical environments** (homes, clinics without EEG labs)
- **Real-time processing** (feedback during acquisition)

### MVP Roadmap

**Phase 1 (Months 1-3): Cloud API**
- FastAPI endpoint for EEG processing
- AWS/GCP deployment
- Integration with hospital PACS systems
- Authentication & audit logging

**Phase 2 (Months 4-6): Clinical Validation**
- IRB approval for pilot study (1 hospital partner)
- 50-100 patient dataset collection
- Cross-validation with clinical neuropsychological assessments
- FDA regulatory strategy development

**Phase 3 (Months 7-12): Regulatory & Scale**
- FDA 510(k) submission (predicate device: standard EEG)
- OEM hardware partnerships (Muse, EMOTIV, g.tec)
- Multi-center clinical trials (3-5 hospital systems)

## Research Background

This algorithm was developed based on:

- **Signal Processing**: Wiener filtering theory (Wiener, 1949)
- **EEG Neuroscience**: Delta-band biomarkers in cognitive decline (Poil et al., 2013)
- **Hardware Design**: Constraints of portable dry-electrode EEG systems
- **Clinical Validation**: 100-sample semi-synthetic dataset mimicking real noise conditions

## File Structure

```
cerebro-abnc/
├── README.md                    # This file
├── LICENSE                      # MIT License
├── requirements.txt             # Python dependencies
├── abnc_algorithm.py            # Core denoising algorithm
├── examples/
│   └── full_pipeline.py        # Complete example with data generation
├── docs/
│   ├── index.html              # Landing page
│   ├── style.css               # Website styling
│   └── images/
│       ├── abnc_waveforms.png  # Before/after visualization
│       └── abnc_frequency.png  # Frequency domain analysis
└── data/
    └── metrics.json            # Performance metrics
```

## Citation

If you use ABNC in your research, please cite:

```bibtex
@software{rahin2026abnc,
  author = {Rahin, Saima Ahmed},
  title = {ABNC: Active Brain Noise Cancellation for Portable EEG},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/saimaahmedrahin/cerebro-abnc}}
}
```

## License

MIT License - See LICENSE file for details

## Contact

**Cerebro Solutions**
- CEO/CIO: Saima Ahmed Rahin
- Email: rahinahmed8@gmail.com
- GitHub: [@saimaahmedrahin](https://github.com/saimaahmedrahin)

---

**Status**: MVP / Proof-of-Concept (Not FDA-approved, for research use only)
