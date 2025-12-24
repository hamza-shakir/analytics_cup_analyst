# SkillCorner × PySport Analytics Cup - Analyst Track Submission

### Gamestate Tactical Analytics Toolkit

**Author:** Hamza Adhnan Shakir

---

## Introduction

When the scoreline changes, tactics change. But how?

Taking the lead provides structural advantage—teams can sit deeper, absorb pressure, exploit transitions. Conceding forces urgency—opponents push higher, commit bodies forward, accept defensive risk. What's overlooked is the duality: both sides adapt simultaneously.

The team that scores faces a choice: maintain aggression or shift to game management? The trailing team confronts mounting pressure: maintain structure or abandon shape in desperation? Responses vary dramatically.

Standard analytics miss this. Whole-match averages treat 90 minutes as uniform. Even basic game state splits barely scratch the surface.

This toolkit provides granularity: **in-possession vs out-of-possession phases**, **time-segmented analysis**, **score-differential contexts**. Instant segmentation revealing how teams *actually* respond under different contexts.

---

## Use Case(s)

**Pre-Match Opponent Analysis:**  
Build tactical profiles across game states. How do they defend when trailing? Does their line push higher at 0-2 versus 0-1? Revealing exploitable patterns.

**Post-Match Team Review:**  
Evaluate execution across game states. Did the team maintain compactness after conceding? Identify coached responses versus reactive chaos.

**Comparative Preparation:**  
Match opponent vulnerabilities to your strengths. Target specific game state contexts for preparation.

---

## Potential Audience

**Coaches & Analysts:** Data-driven evidence of shape evolution and tactical adjustments across game states.

**Researchers:** Standardized frameworks for studying tactical decision-making and momentum effects.

---

## Video URL

**[video](https://www.loom.com/share/9f77e4a51e5643608f91c9705aa66153)**

---

## Run Instructions

### Prerequisites
Python >= 3.9

### Installation
```bash
git clone https://github.com/hamza-shakir/analytics_cup_analyst.git
cd analytics_cup_analyst
pip install -r requirements.txt
pip install -e .
```

### Verification
```python
python -c "import gamestate as gs; print('✅ Installation successful')"
```

### Running
```bash
jupyter notebook submission.ipynb
```

---