# ReasonMesh

**ReasonMesh** is an open-source repository that implements state-of-the-art reasoning techniques for Large Language Models (LLMs). The goal is to provide a modular and extensible framework for **benchmarking, experimenting, and improving reasoning capabilities** in LLMs.

This project is inspired by the paper:  
[**"An Overview and Discussion on Using Large Language Models for Implementation Generation of Solutions to Open-Ended Problems"**](https://arxiv.org/abs/2501.00562)  

![GitHub visitors](https://visitor-badge.laobi.icu/badge?page_id=lukecage0.ReasonMesh)

---

## 🚧 Project Status: In Development
This project is still under active development. Many features are being implemented, and contributions are welcome to help shape its future.

---

## 📌 Features  
- Implementations of **key reasoning techniques** from recent literature  
- Benchmarking on standard datasets (**GSM8K, AQuA, etc.**)  
- Modular design for **customizing prompt-based reasoning strategies**  
- **Local execution** without relying on API-based models  
- Open-source and **community-driven**  

---

## 📂 Project Structure  
```
ReasonMesh/
├── README.md                   # Project documentation
├── benchmark/                   # Benchmarking datasets and evaluation scripts
│   ├── __init__.py
│   ├── aqua.py                  # AQuA dataset benchmarking
│   ├── gsm8k.py                 # GSM8K dataset benchmarking
│   └── __pycache__/
├── datasets.txt                  # List of datasets used
└── prompt_based_reasoning/       # Prompt-based reasoning methods
    ├── __init__.py
    ├── prompt_templates/         # Various prompt-based templates
    │   ├── prompt_builder.py     # Template for constructing prompts
    ├── setup_and_search_algorithms/
    │   ├── reasoning_engine.py   # Core reasoning engine
    │   ├── self_consistency.py   # Self-consistency reasoning approach
    ├── utils/
    │   ├── model_loader.py       # Utility for loading models
    │   ├── tokenizer.py          # Tokenizer utilities
    └── __pycache__/
```

---

## 🚀 Installation & Setup  
### **Prerequisites**  
- Python 3.8+  
- Virtual environment recommended  

### **1️⃣ Clone the Repository**  
```bash
git clone https://github.com/lukecage0/ReasonMesh.git
cd ReasonMesh
```

### **2️⃣ Install Dependencies**  
```bash
pip install -r requirements.txt
```

### **3️⃣ Run a Benchmark Test**  
Example: Running GSM8K reasoning test  
```bash
python benchmark/gsm8k.py
```

---

## 📊 Benchmarks  
| Dataset | Accuracy | Model |
|---------|----------|--------|
| GSM8K   | TBD      | TBD    |
| AQuA    | TBD      | TBD    |

---

## 🤝 Contributing  
Contributions are welcome! If you’d like to improve ReasonMesh, follow these steps:  

1. Fork the repository  
2. Create a feature branch (`git checkout -b feature-name`)  
3. Commit your changes (`git commit -m "Add new feature"`)  
4. Push to your fork (`git push origin feature-name`)  
5. Open a Pull Request  

---

## 📜 License  
This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## 📩 Contact  
For discussions, issues, or contributions, feel free to open an [issue](https://github.com/lukecage0/ReasonMesh/issues) or reach out.  

---

## ✅ Next Steps  
- Add more datasets for benchmarking  
- Implement additional reasoning techniques  
- Optimize model execution efficiency  
- Community contributions for extending features  

---

