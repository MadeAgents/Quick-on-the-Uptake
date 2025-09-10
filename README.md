# Quick on the Uptake

Research code for the paper "Quick on the Uptake: Eliciting Implicit Intents from Human Demonstrations for Personalized Mobile-Use Agents".

Paper link: [https://arxiv.org/abs/2508.08645](https://arxiv.org/abs/2508.08645)
## 🚀 Quick Start
### 1. Environment Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/MadeAgents/Quick-on-the-Uptake
   ```
2. Navigate into the project directory:
   ```bash
   cd Quick-on-the-Uptake
   ```
3. MobileIAR dataset link: [Google Drive]()

   Please download the data and save it as Trajectories.

4. Model link：Warmed-up query rewriter [https://huggingface.co/wuuuuuz/IFRAgent](https://huggingface.co/wuuuuuz/IFRAgent)

(Steps 2 to 4 can be skipped as we have already prepared the personalized query and personalized SOP in the json file of the MobileIAR dataset.)

### 2. Get explicit intent flow and implicit intent flow
```bash
python get_intent_flow.py
```

### 3. Get explicit intent flow vector database
```bash
python get_rag.py
```

### 4. Get personalized query and personalized SOP 
```bash
python rewriting_loop.py
```

### 5. Obtain experimental results from the paper
```bash
python test_loop_{xxx}.py
```
Here, `xxx` refers to different baselines. If you set `test_func=test_loop_{xxx}`, you will get the experimental results of the baseline. If you set `test_func=test_loop_IFRAgent`, you will get the experimental results of `xxx` after being processed by IFRAgent.

## 📑 Citation

Please cite our [paper](https://arxiv.org/abs/2508.08645) if you use this toolkit:

```
@article{wu2025quick,
  title={Quick on the Uptake: Eliciting Implicit Intents from Human Demonstrations for Personalized Mobile-Use Agents},
  author={Wu, Zheng and Huang, Heyuan and Yang, Yanjia and Song, Yuanyi and Lou, Xingyu and Liu, Weiwen and Zhang, Weinan and Wang, Jun and Zhang, Zhuosheng},
  journal={arXiv preprint arXiv:2508.08645},
  year={2025}
}
``` 



