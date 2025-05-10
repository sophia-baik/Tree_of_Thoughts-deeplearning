# Tree of Thoughts

## Introduction
LLMs struggle with structured, multi-step reasoning, limiting performance on complex problems that require planning or exploring multiple solution paths. To address this, we reproduce the Tree of Thoughts (ToT) [4] framework introduced by Yao et al. in 2023, which enhances the Chain of Thought (CoT) [3] approach by enabling language models to generate and explore diverse intermediate “thoughts” through a structured branching process, improving problem solving.

## Chosen Result
We reproduce the results from Section 4.1 of Yao et al’s 2023 paper, which demonstrates the improved performance of ToT on the Game of 24; a game that involves combining four numbers with basic arithmetic operations to create 24, a task that requires strategic planning and logical reasoning. The key result we aim to replicate is paper's ToT accuracy of 74% with b = 5 where b is the maximum breadth of the tree at any given level.

## GitHub Contents
The files in our Github includes all necessary codes and results used for replicating the study,presentation, and results. 
| File Name | Description |
|-----------|-------------|
| Code      | Main implementation files. `/archive` contains unused code from the replication process |
| Data      | Generated input data for Game of 24 and pretrained weights used in the reinforcement learning model |
| Poster    | Presentation poster |
| Report    | Two-page final report |
| Results   | Graphs and tables showing experiment outcomes|


## Re-implementation Details

We created a **dataset** of 1,362 solvable four-number combinations from the range 1–13 using `code/create_dataset_go24.py`. Each combination was categorized into five difficulty levels based on the number of possible solutions.

Our re-implementation includes **five ToT models with different evaluators**:
- 5-shot GPT baseline
- LLM evaluator
- LLM evaluator with backtracking
- Learned evaluator
- Learned evaluator with backtracking

Due to **API, cost, and time constraints**, we made several modifications to the original paper:
- Used **GPT-4o-mini** via OpenAI API instead of GPT-4 (2023), resulting in an implementation ~150× cheaper.
- Introduced a **learned value function**, a neural network trained to estimate the probability of reaching 24 from a given state.
- Replaced full **BFS tree exploration** with a **backtracking mechanism** for efficiency.
- Skipped the evaluation step after step 2, since it involves checking whether 24 can be formed from two remaining numbers—a task that is computationally trivial.

These changes preserved the core logic of Tree of Thoughts while improving practicality for constrained settings.

<img src="results/tot_architecture.png" width="600"/>

## Reproduction Steps

1. Create a .env file and add your api key in the format OPENAI_API_KEY="YOUR_API_KEY_HERE".
2. Simply running following files will return an accuracy score for each evaluation model and the results will be printed on the console: 
   - LLM evaluation: '/code/run_not_iid_tot_go24.py'
   - LLM with backtracking: '/code/run_backtracking_tot_go24.py'
   - Learned evaluation: '/code/run_value_net_tot_go24.py'
   - Baseline 5-shots learning: '/code/run_zeroshot_go24.py'
3. Adjust the dataset range in the evaluation model files to test specific difficulty level.



## Results/Insights
All of our ToT models outperform the baseline 5-shot GPT across all difficulty levels, as shown in the figure below. Incorporating backtracking achieves accuracy comparable to the original paper, following the performance hierarchy: LLM evaluator > Learned evaluator > 5-shot.

<img src="results/accuracy_plot.png" width="600"/>

 Notably, our ToT model with the LLM evaluator is also approximately 150 times more cost-efficient than the original implementation.
 
<img src="results/thetable.png" width="500"/>


## Conclusion
Our reproduction confirmed that ToT outperforms traditional LLMs on structured reasoning tasks like G24. The
systematic exploration of reasoning paths and backtracking are key to performance gains for G24, and more
broadly, complex reasoning tasks.

## References
[1] Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization, 2017.

[2] OpenAI. Openai api. https://platform.openai.com, 2024. Computer software.

[3] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, and
Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models, 2023.

[4] Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, and Karthik
Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. 2023.

## Acknowledgements
This was the final project for Cornell University's CS 4/5782 Introduction to Deep Learning.
