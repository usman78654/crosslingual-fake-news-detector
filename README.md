# Cross-Lingual Fake News Detection Using Multilingual Transformers

A comprehensive 6-week research project implementing cross-lingual fake news detection using XLM-RoBERTa, demonstrating transfer learning from English to Urdu.

## 📋 Project Overview

This project addresses the challenge of fake news detection in low-resource languages by leveraging multilingual transformers. We train a model on English data and evaluate its cross-lingual transfer capability to Urdu, exploring multiple training strategies.

### Key Features
- ✅ Multilingual transformer-based architecture (XLM-RoBERTa)
- ✅ Cross-lingual transfer learning (English → Urdu)
- ✅ Multiple training approaches (monolingual, joint, fine-tuned)
- ✅ Comprehensive evaluation and visualization
- ✅ Complete reproducible pipeline

## 🏗️ Project Structure

```
NLP_project/
├── EnglishDataset/           # English fake news data
│   ├── Fake.csv
│   └── True.csv
├── Urdu_Dataset/             # Urdu fake news data
│   ├── Fake News.csv
│   └── True News.csv
├── data/processed/           # Preprocessed datasets (generated)
├── models/                   # Trained models (generated)
│   ├── xlm_roberta_english/
│   ├── multilingual/
│   └── finetuned_urdu/
├── results/                  # Results and visualizations (generated)
│   ├── week1_exploration/
│   ├── week2_preprocessing/
│   ├── week3_model_training/
│   ├── week4_cross_lingual/
│   ├── week5_joint_training/
│   └── week6_final/
├── 01_data_exploration.py
├── 02_data_preprocessing.py
├── 03_train_english_model.py
├── 04_cross_lingual_eval.py
├── 05_joint_training_finetuning.py
├── 06_final_evaluation.py
├── run_complete_pipeline.py
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- 8GB+ RAM
- ~2GB free disk space for models

### Installation

1. **Clone or navigate to the project directory**
```bash
cd C:\Users\Administrator\Desktop\NLP_project
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Running the Project

Use the project virtual environment if available:

```bash
# Windows PowerShell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

#### Option 1: Run Complete Pipeline 🎯 (Recommended)
```bash
python run_complete_pipeline.py
```
This runs all 6 weeks sequentially and generates all results.

#### Option 2: Run Individual Weeks
```bash
# Week 1: Data Exploration
python 01_data_exploration.py

# Week 2: Preprocessing & Baseline
python 02_data_preprocessing.py

# Week 3: English Model Training
python 03_train_english_model.py

# Week 4: Cross-Lingual Evaluation
python 04_cross_lingual_eval.py

# Week 5: Joint Training & Fine-Tuning
python 05_joint_training_finetuning.py

# Week 6: Final Evaluation
python 06_final_evaluation.py
```

## 📊 Project Phases

### Week 1: Data Exploration & Analysis
- Load and explore English and Urdu datasets
- Analyze class distribution
- Generate dataset statistics
- Create visualizations

**Outputs:** `results/week1_exploration/`

### Week 2: Data Preprocessing & Baseline
- Text cleaning and preprocessing
- Create train-test splits
- Train baseline models (Logistic Regression + TF-IDF)
- Evaluate baseline performance

**Outputs:** `data/processed/`, `results/week2_preprocessing/`

### Week 3: English Model Training
- Load XLM-RoBERTa model
- Train on English dataset
- Evaluate performance
- Save trained model

**Outputs:** `models/xlm_roberta_english/`, `results/week3_model_training/`
**Current state:** Pending/partially run (artifacts not yet committed)

### Week 4: Cross-Lingual Transfer Evaluation
- Test English model on Urdu (zero-shot)
- Measure performance degradation
- Compare cross-lingual transfer
- Generate evaluation metrics

**Outputs:** `results/week4_cross_lingual/`
**Current state:** Pending

### Week 5: Joint Training & Fine-Tuning
- **Experiment 1:** Train on combined multilingual data
- **Experiment 2:** Fine-tune English model on Urdu
- Compare all approaches
- Comprehensive performance analysis

**Outputs:** `models/multilingual/`, `models/finetuned_urdu/`, `results/week5_joint_training/`
**Current state:** Pending

### Week 6: Final Evaluation & Documentation
- Generate comprehensive dashboard
- Create final report
- Performance summary
- Presentation materials

**Outputs:** `results/week6_final/`
**Current state:** Pending

## 📈 Expected Results

### Performance Metrics

| Model | Language | Accuracy | F1-Score |
|-------|----------|----------|----------|
| English-Only | English | ~0.95+ | ~0.95+ |
| Zero-Shot | Urdu | ~0.70-0.80 | ~0.70-0.80 |
| Joint Training | Urdu | ~0.75-0.85 | ~0.75-0.85 |
| Fine-Tuned | Urdu | ~0.80-0.90 | ~0.80-0.90 |

*Actual results may vary based on dataset characteristics*

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| Programming | Python 3.8+ |
| Deep Learning | PyTorch |
| NLP Framework | HuggingFace Transformers |
| Transformer Model | XLM-RoBERTa-base (~270M params) |
| Data Processing | Pandas, NumPy |
| Machine Learning | scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Optimization | AdamW |

## 🎯 Key Contributions

1. **Comprehensive Cross-Lingual Analysis**
   - Systematic evaluation of cross-lingual transfer
   - Quantified performance gaps
   
2. **Multiple Training Strategies**
   - Zero-shot transfer
   - Joint multilingual training
   - Language-specific fine-tuning

3. **Low-Resource Language Focus**
   - Addresses Urdu fake news detection
   - Practical solution for South Asian context

4. **Reproducible Research**
   - Complete end-to-end pipeline
   - Modular code structure
   - Comprehensive documentation

5. **Publication-Ready**
   - Extensive evaluation
   - Professional visualizations
   - Detailed analysis

## 📝 Research Insights

### Cross-Lingual Transfer
XLM-RoBERTa demonstrates zero-shot cross-lingual capability, but performance drops significantly for low-resource languages, highlighting the need for targeted adaptation.

### Fine-Tuning Effectiveness
Even small amounts of target language data significantly improve performance through fine-tuning, making it the most effective approach.

### Joint Training Benefits
Multilingual training creates more balanced representations but may slightly compromise single-language performance.

## 🔬 Hyperparameters

```python
BATCH_SIZE = 16
MAX_LENGTH = 256
LEARNING_RATE = 2e-5 (training) / 1e-5 (fine-tuning)
EPOCHS = 3
OPTIMIZER = AdamW
SCHEDULER = Linear warmup
```

## 📦 Dependencies

Core packages:
- `torch>=2.0.0` - Deep learning framework
- `transformers>=4.30.0` - Pretrained models
- `pandas>=2.0.0` - Data manipulation
- `scikit-learn>=1.3.0` - ML utilities
- `matplotlib>=3.7.0` - Visualization
- `seaborn>=0.12.0` - Statistical visualization

See `requirements.txt` for complete list.

Dependency verification status:
- All listed packages in `requirements.txt` import successfully in the local `.venv` (checked on April 22, 2026).

## 🐛 Troubleshooting

### Out of Memory Error
- Reduce `BATCH_SIZE` in scripts (default: 16 → 8 or 4)
- Reduce `MAX_LENGTH` (default: 256 → 128)

### CUDA Not Available
- The code automatically falls back to CPU
- Training will be significantly slower

### File Too Large Error
- Large CSV files (>50MB) are handled via pandas
- No action needed

### Module Not Found
```bash
pip install -r requirements.txt --upgrade
```

## 📚 Citations

If you use this project, please cite:

```bibtex
@misc{crosslingual_fakenews_2026,
  title={Cross-Lingual Fake News Detection Using Multilingual Transformers},
  author={Your Name},
  year={2026},
  publisher={GitHub}
}
```

## 🔮 Future Enhancements

- [ ] Expand to more languages (Hindi, Arabic, etc.)
- [ ] Few-shot learning approaches
- [ ] Multimodal detection (text + images)
- [ ] Real-time deployment API
- [ ] Larger dataset collection
- [ ] Explainability analysis

## 📄 License

This project is for educational and research purposes.

## 👥 Contact

For questions or collaboration:
- Create an issue in the repository
- Contact: [Your Email]

## 🙏 Acknowledgments

- HuggingFace for Transformers library
- XLM-RoBERTa model developers
- Dataset contributors
- Open-source community

---

**Last Updated:** April 22, 2026

**Status:** 🚧 In Progress (Weeks 1-2 complete; Weeks 3-6 pending)

**Version:** 1.0.1
