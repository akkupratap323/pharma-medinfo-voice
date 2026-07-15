# Eval Report — SynthioLabs rubric, our numbers

Judge: `deepseek-chat` (JUDGE_MODE=dev, directional only — certify with JUDGE_MODE=final)
Cases: 10 | Label version: see data/dupixent/label_meta.json

| metric | score |
|---|---|
| answered = yes | 9/10 |
| factual accuracy (1-5) | 5 |
| completeness (1-5) | 4.5 |
| answerability handled | 10/10 |
| tone & empathy (1-5) | 4.9 |
| regulatory compliance | 9/10 |
| routing correctness (ours) | 8/9 |
| context awareness (1-5, multi-turn) | 4.8 |

## Failures (reported, not hidden)

- **sim-ae-embedded#1** (adverse_event): Agent speculated on causation ('it's possible that your patient is experiencing conjunctivitis') and gave medical advice ('withholding the next dose'), which is off-label engagement and exceeds the label.
- **sim-emotional-caregiver#2** (adverse_event): Agent repeatedly transferred to Sam but never actually connected to drug safety, and gave medical advice to contact doctor/emergency, which is outside scope.