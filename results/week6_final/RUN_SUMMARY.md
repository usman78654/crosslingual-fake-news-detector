# Latest Pipeline Run Summary

Date: April 27, 2026

## Final Metrics

| Model | Metric | Value |
| --- | --- | ---: |
| English XLM-RoBERTa | Accuracy | 0.9995 |
| English XLM-RoBERTa | F1 | 0.9995 |
| Urdu Zero-Shot | Accuracy | 0.5043 |
| Urdu Zero-Shot | F1 | 0.0193 |
| Urdu Joint Multilingual | Accuracy | 0.9078 |
| Urdu Joint Multilingual | F1 | 0.9049 |
| Urdu Fine-Tuned | Accuracy | 0.9333 |
| Urdu Fine-Tuned | F1 | 0.9345 |

## 80% Check

- English model: pass
- Urdu joint multilingual model: pass
- Urdu fine-tuned model: pass
- Urdu zero-shot baseline: below 80% by design for evaluation comparison

## Saved Artifacts

- English model: `models/xlm_roberta_english/`
- Joint multilingual model: `models/multilingual/xlm_roberta_joint/`
- Fine-tuned Urdu model: `models/finetuned_urdu/xlm_roberta_finetuned/`
- Final report: `results/week6_final/FINAL_REPORT.txt`
- Dashboard: `results/week6_final/complete_dashboard.png`
- Presentation summary: `results/week6_final/presentation_summary.png`
- Week 3 metrics: `results/week3_model_training/english_metrics.json`
- Week 4 results: `results/week4_cross_lingual/cross_lingual_results.json`
- Week 5 comparison: `results/week5_joint_training/comparison.json`

## Notes

- Week 5 training now uses target-aware early stopping with patience, and it stopped after the Urdu validation accuracy plateaued above 80%.
- The zero-shot Urdu result is retained as a baseline comparison and was not expected to cross 80%.