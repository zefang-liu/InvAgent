# :robot: InvAgent: A LLM-based Multi-Agent System for Inventory Management in Supply Chains

InvAgent is a novel approach leveraging large language models (LLMs) to manage multi-agent inventory systems. It enhances resilience and improves efficiency across the supply chain network through zero-shot learning capabilities, enabling adaptive and informed decision-making without prior training. For more detailed information, please check our paper.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/zefang-liu/InvAgent.git
   cd InvAgent
   ```

2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running Experiments

- To run the AutoGen experiments, use `notebooks/autogen.ipynb`. Note that an `OPENAI_API_KEY` is required as an environment variable.

### Source Code

- The main environment setup is found in `src/env.py`.
- Configure the environment settings in `src/config.py`.
- Implement custom inventory management policies in `src/baseline.py`.
- For specific implementations of IPPO and MAPPO, refer to `src/ippo.py` and `src/mappo.py`, respectively.

## Citation

If you find this repository useful in your research, please consider citing the paper:

```
```

## Contact Us

For more information or any inquiries, please don't hesitate to contact  [yquan9@gatech.edu](mailto:yquan9@gatech.edu) or [liuzefang@gatech.edu](mailto:liuzefang@gatech.edu).

## License

This project is licensed under the Apache-2.0 license. See the [LICENSE](LICENSE) file for details.
